# simple python CLI that takes a url as an argument and a question as input
# uses scrapy to scrape the url
# for the main content of each page, splits content into paragraphs
# if number of tokens in the paragraph (as determined by tiktoken) exceeds MAX_TOKENS, split paragraph in half by sentences recursively until sub-paragraph <= MAX_TOKENS
# create index of (id, (page, paragraph), embedding) in a dataframe
# index the (embedding => id) in AnnoyIndex
# Take a question as input from the terminal
# find the top N_CONTEXT similar paragraphs using an embedding of the question and nearest neighbors in AnnoyIndex
# print answer the question using openai's API and the context

#import necessary libraries 
import argparse
import logging
import os
import re
from urllib.parse import urlparse
from more_itertools import chunked

import pandas as pd
import scrapy
from annoy import AnnoyIndex
from openai import OpenAI
from scrapy.crawler import CrawlerProcess
import tiktoken


# create openai client
openai = OpenAI()

MODEL_VERSION = 'gpt-3.5-turbo'

# set the maximum number of tokens based on the openai model
EMBEDDINGS_VERSION = 'text-embedding-3-small'
MAX_TOKENS = 8191
EMBEDDING_DIM = 1536

ANNOY_TREES = 10

# set the number of nearest neighbors to search in the AnnoyIndex
N_NEIGHBORS = 5

# function that takes the url to scrape and returns the dataframe of paragraphs
def scrape_url(domain, url):
    # create a custom spider to scrape the url
    feed_uri = f'{domain}-output.json'

    class MySpider(scrapy.Spider):
        name = 'myspider'
        start_urls = [url]

        def parse(self, response):
            # extract the main content of the page
            content = []
            try:
                content = (el.xpath('string()').get().strip()
                        for el in response.xpath('//body//p[not(ancestor::script or ancestor::style)]'))
            except scrapy.exceptions.NotSupported:
                logging.warning(f'Warning: unsupported content type: {response.url}')
                return

            encoding = tiktoken.encoding_for_model(EMBEDDINGS_VERSION)
            # split the content into paragraphs
            paragraphs = []
            for text in content:
                text = re.sub(r'\s+', ' ', text).strip()
                if not text:
                    continue

                # use tiktoken to count the number of tokens in the paragraph
                if len(encoding.encode(text)) > MAX_TOKENS:
                    sentences = text.split('.')
                    if len(text) <= 1:
                        sentences = text.split()
                    if len(sentences) <= 1:
                        logging.warning(f'Warning: paragraph too long to split: {text}')
                        continue
                    content.extend(sentences)
                    continue
        
                paragraphs.append(text)
        
            if paragraphs:
                yield {
                    'page': response.url,
                    'paragraphs': paragraphs
                }
            
            # follow only internal links within the domain
            for link in response.css('a::attr(href)').getall():
                if link.startswith('/'):
                    yield response.follow(link, self.parse)
        
    if os.path.exists(feed_uri):
        logging.info(f'{feed_uri} already exists, skipping scraping')
        return True

    # run the spider to scrape the url
    process = CrawlerProcess(settings={
        'LOG_LEVEL': 'INFO',
        'FEEDS': {
            feed_uri: {
                'format': 'json',
                'overwrite': True,
                'encoding': 'utf8',
                'store_empty': False,
                'fields': None,
                'indent': 4,
            }
        }
    })
    process.crawl(MySpider)
    process.start()
    return True # TODO check if scraping was successful

def load_scraped_paragraphs(domain):
    stored_uri = f'{domain}-output.parquet'
    feed_uri = f'{domain}-output.json'

    #try to load compressed version of the dataframe from stored file
    try:
        df = pd.read_parquet(stored_uri)
        return df
    except:
        logging.info(f'{stored_uri} not found, creating dataframe from {feed_uri}')

        # read all of the paragraphs from the output file into a dataframe
        df = pd.read_json(feed_uri, lines=False, orient='records')
        # label the columns of df
        # df.columns = ['page', 'paragraphs']
        # create a new dataframe with a row for each paragraph
        df = df.explode('paragraphs', ignore_index=True)

        # save compressed version of the dataframe
        df.to_parquet(stored_uri, compression='gzip')
        logging.info(f'Saved {len(df)} paragraphs to {stored_uri}')

        return df

def load_annoy_index(url, df):
    index_uri = f'{url}-annoy_index.ann'

    #try to load the AnnoyIndex from stored file
    try:
        annoy_index = AnnoyIndex(EMBEDDING_DIM)
        annoy_index.load(index_uri)
        return annoy_index
    except:
        logging.info('annoy_index.ann not found, creating AnnoyIndex from embeddings')

        # create AnnoyIndex of the OpenAI embeddings of the paragraphs
        annoy_index = AnnoyIndex(EMBEDDING_DIM)
        for chunk in chunked(enumerate(df['paragraphs']), 10):
            idxs, paragraphs = zip(*chunk)
            embeddings = openai.embeddings.create(input=paragraphs, model=EMBEDDINGS_VERSION).data
            for i, r in zip(idxs, embeddings):
                annoy_index.add_item(i, r.embedding)
        annoy_index.build(ANNOY_TREES)
        annoy_index.save(index_uri)
        return annoy_index
    
# function that takes a question and returns the answer
def answer_question(domain, question, df, annoy_index):
    # get the embeddings of the question
    question_embedding = openai.embeddings.create(input=[question], model=EMBEDDINGS_VERSION).data[0].embedding
    # find the nearest neighbors of the question in the AnnoyIndex
    nearest_neighbors = annoy_index.get_nns_by_vector(question_embedding, N_NEIGHBORS)
    # get the paragraphs of the nearest neighbors
    paragraphs = df.iloc[nearest_neighbors]
    # get the context of the paragraphs
    context = '\n'.join(paragraphs['paragraphs'])
    # get the answer to the question using the context
    answer = openai.chat.completions.create(
        model=MODEL_VERSION,
        messages=[
            {'role': 'system', 'content': f'you are answering questions about the website {domain}, with the provided context'},
            {'role': 'user', 'content': f'Question: {question} \nContext: {context}'}]
       ).choices[0].message.content

    return f'Answer: {answer}\n\n For more information read:\n' + '\n'.join(f'{c['page']} - {c['paragraphs']}' for c in paragraphs.to_dict(orient='records'))

def question_prompt():
    while True:
        question = input('Ask a question (or press Enter to quit): ')
        if not question:
            return
        yield question


# main function that takes the url and question as arguments
def main():
    parser = argparse.ArgumentParser(description='Scrape a url and answer a question using OpenAI')
    parser.add_argument('url', help='The domain to scrape')

    args = parser.parse_args()
    # extract the domain from the url
    domain = urlparse(args.url).netloc
    logging.basicConfig(level=logging.INFO)

    result = scrape_url(domain, args.url)
    if not result:
        logging.error("Scraping was unsuccessful.")
        return
    df = load_scraped_paragraphs(domain)
    annoy_index = load_annoy_index(domain, df)

    for question in question_prompt():
        answer = answer_question(domain, question, df, annoy_index)
        print(answer)

if __name__ == '__main__':
    main()






