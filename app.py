from flask import Flask, request, jsonify
from flask_cors import CORS
import ollama
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Constants
MODEL_NAME = "mistral:latest"
HF_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_API_TOKEN = "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"


def load_customers():
    """Load customer data from the customers.txt file."""
    customers = {}
    try:
        with open("customers.txt", "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    customer_id, api_key = parts
                    customers[customer_id] = api_key
    except FileNotFoundError:
        print("Customers file not found.")
    return customers


customers_data = load_customers()
print(customers_data)


def get_response_from_ollama(user_message):
    """Get response from Ollama backend."""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": user_message}],
    )
    if response and "message" in response:
        return response["message"]["content"]
    return None


def get_response_from_huggingface(user_message):
    """Get response from Hugging Face Inference Client."""
    client = InferenceClient(HF_MODEL_NAME, token=HF_API_TOKEN)
    output = client.chat.completions.create(
        model=HF_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ],
        stream=False,
        max_tokens=1024,
    )
    if output and output.choices:
        return output.choices[0].message.content
    return None


@app.route("/api/send-message", methods=["POST"])
def send_message():
    try:
        # Extract JSON payload
        data = request.json
        if not data:
            return jsonify({"error": "Invalid request format. JSON payload is required"}), 400

        # Extract customerId, apiKey, and message
        customer_id = data.get("customerId")
        api_key = data.get("apiKey")
        user_message = data.get("message")
        mode = os.getenv("MODE", "local")

        # Validate inputs
        if not customer_id or not api_key:
            return jsonify({"error": "Customer ID or API Key missing"}), 400
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Validate Customer ID and API Key
        valid_api_key = customers_data.get(customer_id)
        if valid_api_key != api_key:
            return jsonify({"error": "Invalid Customer ID or API Key"}), 403

        # Get AI response based on mode
        print("Calling {} backend", mode)
        if mode == "local":
            ai_message = get_response_from_ollama(user_message)
        elif mode == "remote":
            ai_message = get_response_from_huggingface(user_message)
        else:
            return jsonify({"error": "Invalid MODE configuration"}), 500

        # Check AI response
        if ai_message:
            formatted_response = {
                "response": ai_message,
                "format": "markdown",  # Add format indicator
            }
            print("AI Response:", formatted_response)  # Debugging log
            return jsonify(formatted_response)
        else:
            return jsonify({"error": "Error communicating with the AI model"}), 500

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
