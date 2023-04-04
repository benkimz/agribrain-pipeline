import os

from flask import Flask, jsonify, request
from agpipeline import AgriBrainSuperPipeline, AG_KMODEL_NAME

app = Flask(__name__)

super_pipeline = AgriBrainSuperPipeline(AG_KMODEL_NAME)

# Endpoint for generating solutions/ answers
@app.route('/solve', methods=['POST'])
def solve():
    prompt = request.json["prompt"]
    max_length = request.json["max_length"]
    temperature = request.json["temperature"]
    num_return_sequences = request.json["num_return_sequences"]
    
    solution = super_pipeline.solve(
        prompt=prompt, 
        max_length=max_length, 
        num_return_sequences=num_return_sequences, 
        temperature=temperature
    )
    
    return jsonify({"solution": solution})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8085)))