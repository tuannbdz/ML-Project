from openai import OpenAI
client = OpenAI(
    api_key="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",  # Replace with your actual OpenAI API key
)

assistant = client.beta.assistants.retrieve('asst_vW6GljPSXmi5PX5IK8pxYumn')
thread = client.beta.threads.create()
message = client.beta.threads.messages.create(thread_id=thread.id,role='user',content='Gay có phải là bệnh không?')
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
    instructions='Please address the user as Khach hang.'
)
import time
time.sleep(10)
run_status = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=run.id,
)

if (run_status.status == 'completed'):
    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )
    for msg in reversed(messages.data):
        role = msg.role
        content = msg.content[0].text.value
        print(f'{role.capitalize()}: {content}')