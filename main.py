import csv
import json
import random
import openai
import time
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import logging
import colorlog
import os
from dotenv import load_dotenv
load_dotenv()
# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a ColorFormatter for the logger
formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)s: %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)

# Create a console handler and set the formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)

# Extract the book titles
def csv_load(file):
    with open(file, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            yield row[1]

# Embed text with error handling
def embed_with_error_handling(text):
    try:
        embedding = openai.Embedding.create(
            input=text, 
            engine=OPENAI_ENGINE)["data"][0]["embedding"]
        return embedding
    except Exception as e:
        logger.error(f"Error embedding text: {text}. Error: {str(e)}")
        return None

# Set up variables
FILE = 'csv/Questions Master _ ChildOther.csv'
COLLECTION_NAME = 'title_db'
DIMENSION = 1536
with open(FILE, newline='') as f:
    row_count = sum(1 for row in csv.reader(f))
# Use the minimum of row_count and a specified maximum count
COUNT = min(row_count, 100) # Free OpenAI account limited to 100k tokens per month

MILVUS_HOST = os.environ.get('MILVUS_HOST')
MILVUS_PORT = os.environ.get('MILVUS_PORT')
OPENAI_ENGINE = os.environ.get('OPENAI_ENGINE')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

MILVUS_HOST = MILVUS_HOST
MILVUS_PORT =MILVUS_PORT
OPENAI_ENGINE = OPENAI_ENGINE
openai.api_key = OPENAI_API_KEY  # 

# Connect to Milvus
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
logger.info("Connected to Milvus.")

# Remove collection if it already exists
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)
    logger.info(f"Collection '{COLLECTION_NAME}' already exists. Dropped the existing collection.")

# Create collection schema
fields = [
    FieldSchema(name='id', dtype=DataType.INT64, description='Ids', is_primary=True, auto_id=False),
    FieldSchema(name='title', dtype=DataType.VARCHAR, description='Title texts', max_length=200),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description='Embedding vectors', dim=DIMENSION)
]
schema = CollectionSchema(fields=fields, description='Title collection')
collection = Collection(name=COLLECTION_NAME, schema=schema)
logger.info("Created collection schema.")

# Create an index for the collection
index_params = {
    'index_type': 'IVF_FLAT',
    'metric_type': 'L2',
    'params': {'nlist': 1024}
}
collection.create_index(field_name="embedding", index_params=index_params)
logger.info("Created index for the collection.")

# Insert each title and its embedding with error handling
for idx, text in enumerate(random.sample(sorted(csv_load(FILE)), k=COUNT)):
    logger.debug(f"Inserting text '{text}' with index '{idx}'.")
    embedding = embed_with_error_handling(text)
    if embedding is not None:
        ins = [[idx], [(text[:198] + '..') if len(text) > 200 else text], [embedding]]
        try:
            collection.insert(ins)
            logger.debug(f"Text '{text}' inserted successfully.")
            time.sleep(3)  # Free OpenAI account limited to 60 RPM
        except Exception as e:
            logger.error(f"Error inserting text '{text}' into collection. Error: {str(e)}")

# Load the collection into memory for searching
collection.load()
logger.info("Loaded collection into memory for searching.")

# Search text with error handling
def search_with_error_handling(text):
    try:
        logger.debug(f"Searching for text '{text}' in collection.")
        embedded_text = embed_with_error_handling(text)
        if embedded_text:
            search_params = {"metric_type": "L2"}
            results = collection.search(
                data=[embedded_text],
                anns_field="embedding",
                param=search_params,
                limit=5,
                output_fields=['title']
            )
            ret = []
            for hit in results[0]:
                row = [hit.id, hit.score, hit.entity.get('title')]
                ret.append(row)
            return ret
        else:
            return []
    except Exception as e:
        logger.error(f"Error searching for text '{text}' in collection. Error: {str(e)}")
        return []

# Perform searches
search_terms = ['self-improvement', 'landscape']

for x in search_terms:
    logger.info(f"Search term: {x}")
    for result in search_with_error_handling(x):
        logger.info(result)
    logger.info('')
