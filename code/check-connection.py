import os
import openai

# Load your API key from environment variable
api_key = os.getenv('OPENAI_API_KEY')
endpoint = os.getenv('OPENAI_ENDPOINT')

# Configure your OpenAI API key and endpoint
openai.api_key = api_key
openai.api_base = endpoint

def query_openai(prompt):
    try:
        response = openai.Completion.create(
            engine="davinci-codex",  # You can change the engine as needed
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return str(e)

# Example query
response_text = query_openai("Translate the following English text to French: 'Hello, how are you?'")
print(response_text)


setx OPENAI_API_KEY='5926f697acf84353a735fd36c3a2fb5f'
setx OPENAI_ENDPOINT='https://openai-up5qjvvyc7xnk.openai.azure.com/'

