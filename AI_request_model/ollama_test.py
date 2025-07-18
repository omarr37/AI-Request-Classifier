import requests

data = {
    "model": "llama3",
    "prompt": "Hello! Give me a joke."
}

response = requests.post("http://localhost:11434/api/generate", json=data, stream=True)

for line in response.iter_lines():
    if line:
        print(line.decode())
