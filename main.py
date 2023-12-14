import csv
import openai
import time
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# Set up Milvus connection parameters
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'

# Set up OpenAI API key and engine
openai.api_key = 'sk-******'  # Use your own Open AI API Key here
OPENAI_ENGINE = 'text-embedding-ada-002'  # Which engine to use

# Function to extract book titles from CSV
def csv_load(file):
    with open(file, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            yield row[1]

# Function to get embeddings from OpenAI
def embed(text):
    return openai.Embedding.create(input=text, engine=OPENAI_ENGINE)["data"][0]["embedding"]

# Connect to Milvus
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

# Create Milvus collection
COLLECTION_NAME = 'title_db'
if COLLECTION_NAME in connections.list_collections():
    connections.drop_collection(COLLECTION_NAME)

fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True),
    FieldSchema(name='title', dtype=DataType.STRING),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=1536)
]
schema = CollectionSchema(fields=fields, description='Title collection')
collection = Collection(name=COLLECTION_NAME, schema=schema)

# Insert book titles and embeddings into Milvus
FILE = './content/books.csv'
COUNT = 100  # Number of titles to embed and insert

for idx, text in enumerate(csv_load(FILE)):
    if idx >= COUNT:
        break
    embedding = embed(text)
    ins = [[idx], [text], [embedding]]
    collection.insert(ins)
    time.sleep(3)  # Free OpenAI account limited to 60 RPM

# Load collection into memory for searching
collection.load()

# Function to search based on input text
def search(text):
    embedding = embed(text)
    results = collection.search(
        data=[embedding],
        anns_field="embedding",
        param={'metric_type': 'L2'},
        limit=5,
        output_fields=['title']
    )

    ret = []
    for hit in results[0]:
        row = [hit.id, hit.score, hit.entity.get('title')]
        ret.append(row)
    return ret

# Perform search on specific terms
search_terms = ['self-improvement', 'landscape']

for term in search_terms:
    print('Search term:', term)
    for result in search(term):
        print(result)
    print()
