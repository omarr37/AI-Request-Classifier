import joblib
import requests
import json

def classify_request(text, model, vectorizer=None):
    """
    تصنيف نص الطلب باستخدام النموذج مباشرة (Pipeline)
    """
    prediction = model.predict([text])[0]
    return prediction


def generate_agent_response(category):
    responses = {
        "IT Support": "Please check the cables or try restarting the device. Do you want to open a support ticket?",
        "IT Request": "Your IT request has been received. We'll process it shortly.",
        "Cleaning": "A cleaning request has been logged. The cleaning team will be notified immediately.",
        "Coffee Request": "Coffee request received ☕. Someone will bring it to you shortly!",
        "Maintenance": "Maintenance team has been notified. They will follow up soon.",
    }
    return responses.get(category, "Thank you! Your request has been received.")

def ask_ollama(prompt, model="llama3"):
    data = {
        "model": model,
        "prompt": prompt
    }
    response = requests.post("http://localhost:11434/api/generate", json=data, stream=True)
    final_answer = ""
    for line in response.iter_lines():
        if line:
            msg = json.loads(line.decode())
            if "response" in msg:
                final_answer += msg["response"]
            if msg.get("done"):
                break
    return final_answer
