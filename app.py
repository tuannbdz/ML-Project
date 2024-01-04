from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/model', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        ans = input_data['prompt']
        # Your NLP model code here (replace this with your actual model logic)
        prediction = {'ans':ans}
        return jsonify(prediction)

    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True,port=5000)
