from flask import Flask, request, jsonify
import csv
import random
import os
import pandas as pd
import openai
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import logging
import colorlog
from dotenv import load_dotenv
import time
load_dotenv()

app = Flask(__name__)

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
def csv_load(filepath):
  with open(filepath, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            yield row  # Yield entire row
# Embed text with error handling
def embed_with_error_handling(text):
    try:
        embedding = openai.Embedding.create(
            input=text, 
            engine=os.environ.get('OPENAI_ENGINE')
        )["data"][0]["embedding"]
        return embedding  # Ensure this returns a list of numbers
    except Exception as e:
        logger.error(f"Error embedding text: {text}. Error: {str(e)}")
        return None


@app.route('/search', methods=['GET'])
def search():
    search_term = request.args.get('q')

    MILVUS_HOST = os.environ.get('MILVUS_HOST')
    MILVUS_PORT = os.environ.get('MILVUS_PORT')

    # Connect to Milvus
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    logger.info("Connected to Milvus.")

    # Fetch all collections
    collections = utility.list_collections()

    logger.info("Collections in Milvus:")
    search_results_per_collection = {}

    for collection_name in collections:
        logger.info(f'collection_name: {collection_name}')

        # Search text in each collection
        search_results = search_in_collection(collection_name, search_term)

        # Store search results for this collection
        search_results_per_collection[collection_name] = search_results

    return jsonify({"results": search_results_per_collection})


def search_in_collection(collection_name, search_term):
    # Get the collection object
    collection = Collection(collection_name)

    def search_with_error_handling(text):
        try:
            logger.debug(f"Searching for text '{text}' in collection '{collection_name}'.")
            embedded_text = embed_with_error_handling(text)
            if embedded_text:
                search_params = {"metric_type": "L2"}
                results = collection.search(
                    data=[embedded_text],
                    anns_field="embedding",
                    param=search_params,
                    limit=1,
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
            logger.error(f"Error searching for text '{text}' in collection '{collection_name}'. Error: {str(e)}")
            return []

    # Perform searches using only the provided search term
    search_results = search_with_error_handling(search_term)
    return {search_term: search_results} if search_results else {}

# Endpoint to delete a collection
@app.route('/delete_collection', methods=['DELETE'])
def delete_collection():
    collection_name = request.args.get('collection_name')

    # Get Milvus connection parameters
    MILVUS_HOST = os.environ.get('MILVUS_HOST')
    MILVUS_PORT = os.environ.get('MILVUS_PORT')

    # Connect to Milvus
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    
    # Check if the collection exists
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        logger.info(f"Collection '{collection_name}' deleted successfully.")
        return jsonify({"message": f"Collection '{collection_name}' deleted successfully."}), 200
    else:
        return jsonify({"message": f"Collection '{collection_name}' does not exist."}), 404

# Endpoint to get a list of collections
@app.route('/collections', methods=['GET'])
def get_collections():
    # Get Milvus connection parameters
    MILVUS_HOST = os.environ.get('MILVUS_HOST')
    MILVUS_PORT = os.environ.get('MILVUS_PORT')

    # Connect to Milvus
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

    # Fetch all collections
    collections = utility.list_collections()
    logger.info("collections:{collections}}")  
    return jsonify({"collections": collections}), 200

def handle_empty_values(value):
    # Check if the value is empty or null
    if pd.isnull(value) or value == '':
        # If the value is empty or null, replace it with a default string value or handle it based on your use case
        return 'N/A'  # For example, replacing empty values with 'N/A'
    else:
        return str(value) 

def create_collection_schema(header):
    fields = [
    FieldSchema(
        is_primary=True if col_name == 'question_id' else False,
        name=col_name,
        dtype=DataType.INT64 if col_name == 'question_id' else DataType.VARCHAR,
        max_length=1024 if col_name != 'question_id' else None
    )
    for col_name in header
]

    fields.append(FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024))

    return CollectionSchema(fields=fields, description="Dynamic Collection from CSV")


# Extract embedding from text using OpenAI
def GetEmbedding(text):
    return openai.Embedding.create(
        input=text, 
        engine='text-embedding-ada-002')["data"][0]["embedding"]

def process_csv_data(file, collection):
    logger.info(f"Processing CSV data from file: {file}")

    with open(file, newline='') as f:
        row_count = sum(1 for row in csv.reader(f))
        COUNT = row_count# Free OpenAI account limited to 100k tokens per month
        
        for idx, textRow in enumerate(csv_load(file)):
            logger.info(f"Processing row {textRow[0]} of {textRow},{idx}")
            embedding = GetEmbedding(textRow[0])
            headerList=pd.read_csv(file).columns
            ins = [{'id': idx,  'embedding': embedding}]
            data_to_insert = []
                    
            ins = {'id': idx, 'embedding': embedding}
          
            for i, val in enumerate(textRow):
                
                val  # Assuming all columns contain text to be embedded
                # Avoid assigning 'id' and 'embedding' again; consider other columns
                if headerList[i] not in ['id', 'embedding']:
                    ins[headerList[i]] = val  # Assign text to corresponding column in 'ins'
            collection.insert([ins])
                        # data_to_insert.append(ins)
        
        # collection.insert([data_to_insert])
            # if data_to_insert:
            #     try:
            #         collection.insert([data_to_insert])
            #     except Exception as e:
            #         logger.error(f"Error inserting data into collection: {str(e)}")
            #     time.sleep(3) 


@app.route('/create_and_store_data', methods=['POST'])
def create_and_store_data():
    file = 'csv/Questions Master _ ChildOther.csv'
    collection_name = 'QuestionsMaster_ChildOther'

    MILVUS_HOST = os.environ.get('MILVUS_HOST')
    MILVUS_PORT = os.environ.get('MILVUS_PORT')
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

    openai.api_key = OPENAI_API_KEY

    try:
        # Establish Milvus connection
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    except Exception as milvus_conn_error:
        logger.error(f"Milvus connection error: {str(milvus_conn_error)}")
        return jsonify({"error": f"Milvus connection error: {str(milvus_conn_error)}"}), 500

    try:
        # List collections
        collections = utility.list_collections()

        if collection_name not in collections:
            schema = create_collection_schema(pd.read_csv(file).columns)
            collection = Collection(name=collection_name, schema=schema)
            collection.create()
            process_csv_data(file, collection)
            return jsonify({"message": f"Collection '{collection_name}' created and data stored successfully."}), 201
        else:
            collection = Collection(name=collection_name)
            process_csv_data(file, collection)
            return jsonify({"message": f"Collection '{collection_name}' already exists. Data inserted successfully."}), 200
    except Exception as collection_error:
        logger.error(f"Collection creation or insertion error: {str(collection_error)}")
        return jsonify({"error": f"Collection creation or insertion error: {str(collection_error)}"}), 500

    
if __name__ == '__main__':
    app.run(debug=True)
