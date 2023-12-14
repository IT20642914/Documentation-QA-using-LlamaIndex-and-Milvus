# app.py

from flask import Flask, jsonify, request
from milvus_interaction import save_to_milvus, search_in_milvus

app = Flask(__name__)

@app.route('/process_csv', methods=['POST'])
def process_csv():
    # Endpoint to process CSV and save to Milvus
    response = save_to_milvus()
    return jsonify(response)

@app.route('/search', methods=['GET'])
def search():
    text = request.args.get('text')
    results = search_in_milvus(text)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
