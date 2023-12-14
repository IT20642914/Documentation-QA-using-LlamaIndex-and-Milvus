import os
import csv
import openai
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import logging
import colorlog

# Load environment variables or set them directly
MILVUS_HOST = os.environ.get('MILVUS_HOST')
MILVUS_PORT = os.environ.get('MILVUS_PORT')
OPENAI_ENGINE = os.environ.get('OPENAI_ENGINE')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

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

MILVUS_HOST = MILVUS_HOST
MILVUS_PORT = MILVUS_PORT
OPENAI_ENGINE = OPENAI_ENGINE
openai.api_key = OPENAI_API_KEY

# Define 'collection' as a global variable
collection = None

def get_collection_name(file_path):
    # Extract file name without extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    return file_name

def csv_load(file):
    with open(file, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            yield row[1]

def embed_with_error_handling(text):
    try:
        embedding = openai.Embedding.create(
            input=text, 
            engine=OPENAI_ENGINE
        )["data"][0]["embedding"]
        return embedding
    except Exception as e:
        logger.error(f"Error embedding text: {text}. Error: {str(e)}")
        return None

def connect_milvus():
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    logger.info("Connected to Milvus.")

def create_collection(file_path):
    global collection  # Define 'collection' as a global variable to be accessed in other functions
    collection_name = get_collection_name(file_path)
    file_name = os.path.splitext(collection_name)[0]  # Extract file name without extension
    collection_name = ''.join(c if c.isalnum() or c == '_' else '' for c in file_name)
    if not utility.has_collection(collection_name):
        fields = [
            FieldSchema(name='id', dtype=DataType.INT64, description='Ids', is_primary=True, auto_id=False),
            FieldSchema(name='title', dtype=DataType.VARCHAR, description='Title texts', max_length=200),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description='Embedding vectors', dim=1536)
        ]
        schema = CollectionSchema(fields=fields, description='Title collection')
        collection = Collection(name=collection_name, schema=schema)
        index_params = {
            'index_type': 'IVF_FLAT',
            'metric_type': 'L2',
            'params': {'nlist': 1024}
        }
        collection.create_index(field_name='embedding', index_params=index_params)
        logger.info(f"Created collection '{collection_name}' schema and index for the collection.")
        

def save_to_milvus(count=100):
    global collection  # Access the global 'collection' variable
    current_directory = os.getcwd()
    FILE = 'csv/Questions Master _ ChildOther.csv'
    FilePath = os.path.join(current_directory, FILE)
    
    connect_milvus()
    create_collection(FILE)

    for idx, text in enumerate(csv_load(FilePath)):
        if idx >= count:
            break

        embedding = embed_with_error_handling(text)
        if embedding is not None:
            ins = [
                [idx],
                [(text[:198] + '..') if len(text) > 200 else text],
                [embedding]
            ]
            try:
                collection.insert(ins)
                logger.debug(f"Text '{text}' inserted successfully.")
            except Exception as e:
                logger.error(f"Error inserting text '{text}' into collection. Error: {str(e)}")

def search_in_milvus(text):
    global collection  # Access the global 'collection' variable
    connect_milvus()
    results = []
    embedded_text = embed_with_error_handling(text)
    if embedded_text:
        search_params = {"metric_type": "L2"}
        try:
            results = collection.search(
                data=[embedded_text],
                anns_field="embedding",
                param=search_params,
                limit=5,
                output_fields=['title']
            )
        except Exception as e:
            logger.error(f"Error searching for text '{text}' in collection. Error: {str(e)}")

    return results