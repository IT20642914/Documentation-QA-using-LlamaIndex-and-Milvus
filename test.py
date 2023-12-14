from flask import Flask, request, jsonify
import csv
import random
import os
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
            engine=os.environ.get('OPENAI_ENGINE')
        )["data"][0]["embedding"]
        return embedding  # Ensure this returns a list of numbers
    except Exception as e:
        logger.error(f"Error embedding text: {text}. Error: {str(e)}")
        return None

@app.route('/process_file', methods=['POST'])
def process_file():
    current_directory = os.getcwd()
    FILE = 'csv/Questions Master _ ChildOther.csv'  # Update the file path separator to '/'
    FilePath = os.path.join(current_directory, FILE)
    COLLECTION_NAME = 'title_db'
    DIMENSION = 1536

    MILVUS_HOST = os.environ.get('MILVUS_HOST')
    MILVUS_PORT = os.environ.get('MILVUS_PORT')

    # Connect to Milvus
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    logger.info("Connected to Milvus.")

    # Create collection schema and other setup steps...

    # Insert each title and its embedding with error handling
    collection = Collection(name=COLLECTION_NAME)  # Create collection object
   # Assuming your Milvus collection expects three fields: 'id', 'title', 'embedding'
    for idx, text in enumerate(csv_load(FilePath)):
        logger.debug(f"Inserting text '{text}' with index '{idx}'.")
        embedding = embed_with_error_handling(text)
        if embedding is not None:
            ins = [
                {'id': idx, 'title': text[:198] if len(text) > 200 else text, 'embedding': embedding}
            ]
            try:
                collection.insert(ins)
                logger.debug(f"Text '{text}' inserted successfully.")
                time.sleep(3)  # Free OpenAI account limited to 60 RPM
            except Exception as e:
                logger.error(f"Error inserting text '{text}' into collection. Error: {str(e)}")

    # Load the collection into memory for searching
    collection.load()
    logger.info("Loaded collection into memory for searching.")

    return jsonify({"message": "File processed and data inserted into the collection."})


# @app.route('/search', methods=['GET'])
# def search():
#     search_term = request.args.get('q')

#     MILVUS_HOST = os.environ.get('MILVUS_HOST')
#     MILVUS_PORT = os.environ.get('MILVUS_PORT')

#     # Connect to Milvus
#     connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
#     logger.info("Connected to Milvus.")
   

#     # Fetch all collections
#     collections = utility.list_collections()

#     logger.info("Collections in Milvus:")
#     for collection_name in collections:
#         logger.info(f'collection_name: {collection_name}')
#     def search_with_error_handling(text):
#         try:
#             logger.debug(f"Searching for text '{text}' in collection.")
#             embedded_text = embed_with_error_handling(text)
#             if embedded_text:
#                 search_params = {"metric_type": "L2"}
#                 results = collection.search(
#                     data=[embedded_text],
#                     anns_field="embedding",
#                     param=search_params,
#                     limit=5,
#                     output_fields=['title']
#                 )
#                 ret = []
#                 for hit in results[0]:
#                     row = [hit.id, hit.score, hit.entity.get('title')]
#                     ret.append(row)
#                 return ret
#             else:
#                 return []
#         except Exception as e:
#             logger.error(f"Error searching for text '{text}' in collection. Error: {str(e)}")
#             return []

#     # Perform searches
#     search_terms = [search_term] if search_term else ['self-improvement', 'landscape']

#     search_results = {}
#     for term in search_terms:
#         results = search_with_error_handling(term)
#         search_results[term] = results

#     return jsonify({"results": search_results})
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
if __name__ == '__main__':
    app.run(debug=True)
