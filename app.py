from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

client = OpenAI(
    api_key="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",  # Replace with your actual OpenAI API key
)

assistant = client.beta.assistants.retrieve('asst_vW6GljPSXmi5PX5IK8pxYumn')
thread = client.beta.threads.create()

@app.route('/api/model', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        ans = input_data['prompt']

        message = client.beta.threads.messages.create(thread_id=thread.id,role='user',content=ans)
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
            instructions='Please address the user as Khach hang.'
        )

        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )

        while (run_status.status != 'completed'):
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id,
            )
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            ans = messages.data[0].content[0].text.value
        prediction = {'ans':ans}
        return jsonify(prediction)

    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True,port=5000)
