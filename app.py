from flask import Flask, request, jsonify
from flask_cors import CORS
import ollama
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from datetime import datetime

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Data storage
page_contexts = {}  # Store page contexts by session ID
session_customer_map = {}  # Map session IDs to customer IDs

# Constants
MODEL_NAME = "mistral:latest"
HF_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")


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


def format_page_context(page_data):
    """Format page data into a comprehensive context string."""
    context_parts = []

    # Add URL and title
    context_parts.append(f"Page URL: {page_data.get('url', 'N/A')}")
    context_parts.append(f"Page Title: {page_data.get('title', 'N/A')}")

    # Add meta tags information
    meta_tags = page_data.get('metaTags', [])
    if meta_tags:
        meta_desc = next((tag['content'] for tag in meta_tags if tag.get('name') == 'description'), None)
        if meta_desc:
            context_parts.append(f"Meta Description: {meta_desc}")

    # Add headings with their hierarchy
    headings = page_data.get('headings', [])
    if headings:
        context_parts.append("\nPage Structure:")
        for heading in headings:
            level = heading.get('level', '').replace('H', '')
            text = heading.get('text', '')
            context_parts.append(f"{'  ' * (int(level) - 1)}• {text}")

    # Add main content if available
    main_content = page_data.get('mainContent', [])
    if main_content:
        context_parts.append("\nMain Content:")
        context_parts.extend(main_content)

    # Add full text content
    full_text = page_data.get('fullText', '')
    if full_text:
        context_parts.append("\nComplete Page Content:")
        context_parts.append(full_text)

    # Add links if available
    links = page_data.get('links', [])
    if links:
        context_parts.append("\nPage Links:")
        for link in links:
            text = link.get('text', '')
            href = link.get('href', '')
            if text and href:
                context_parts.append(f"• {text}: {href}")

    return "\n".join(context_parts)


def get_response_from_ollama(user_message, context):
    """Get response from Ollama backend with context."""
    system_prompt = f"""You are a helpful assistant. Use the following page context to help answer questions. 
    The context contains the complete content of the webpage the user is viewing:

    {context}

    Please provide accurate answers based on this context. If the information isn't available in the context, 
    you can say so."""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
    )
    if response and "message" in response:
        return response["message"]["content"]
    return None


def get_response_from_remote(user_message, context):
    """Get response from Hugging Face Inference Client with context."""
    client = InferenceClient(HF_MODEL_NAME, token=HF_API_TOKEN)
    system_prompt = f"""You are a helpful assistant. Use the following page context to help answer questions. 
    The context contains the complete content of the webpage the user is viewing:

    {context}

    Please provide accurate answers based on this context. If the information isn't available in the context, 
    you can say so."""

    output = client.chat.completions.create(
        model=HF_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        stream=False,
        max_tokens=1024,
    )
    if output and output.choices:
        return output.choices[0].message.content
    return None


def validate_customer(customer_id, api_key):
    """Validate customer credentials."""
    valid_api_key = customers_data.get(customer_id)
    return valid_api_key == api_key


@app.route('/api/page-context', methods=['POST'])
def store_page_context():
    try:
        data = request.json
        session_id = data.get('sessionId')
        customer_id = data.get('customerId')
        api_key = data.get('apiKey')
        page_data = data.get('pageData')

        if not all([session_id, customer_id, api_key, page_data]):
            return jsonify({'error': 'Missing required fields'}), 400

        # Validate customer credentials
        if not validate_customer(customer_id, api_key):
            return jsonify({'error': 'Invalid Customer ID or API Key'}), 403

        # Store the page context with session ID
        page_contexts[session_id] = {
            'pageData': page_data,
            'timestamp': datetime.now().isoformat(),
            'customerId': customer_id
        }

        # Map session to customer
        session_customer_map[session_id] = customer_id

        return jsonify({'message': 'Page context stored successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/page-context-update', methods=['POST'])
def update_page_context():
    try:
        data = request.json
        session_id = data.get('sessionId')
        customer_id = data.get('customerId')
        api_key = data.get('apiKey')
        page_data = data.get('pageData')

        if not all([session_id, customer_id, api_key, page_data]):
            return jsonify({'error': 'Missing required fields'}), 400

        # Validate customer credentials
        if not validate_customer(customer_id, api_key):
            return jsonify({'error': 'Invalid Customer ID or API Key'}), 403

        # Validate session exists and belongs to customer
        if session_id not in page_contexts or page_contexts[session_id]['customerId'] != customer_id:
            return jsonify({'error': 'Invalid session ID'}), 403

        # Update the page context
        page_contexts[session_id].update({
            'pageData': page_data,
            'timestamp': datetime.now().isoformat()
        })

        return jsonify({'message': 'Page context updated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/api/send-message", methods=["POST"])
def send_message():
    try:
        # Extract JSON payload
        data = request.json
        if not data:
            return jsonify({"error": "Invalid request format. JSON payload is required"}), 400

        # Extract data
        session_id = data.get("sessionId")
        customer_id = data.get("customerId")
        api_key = data.get("apiKey")
        user_message = data.get("message")
        current_url = data.get("currentUrl")
        mode = os.getenv("MODE", "remote")

        # Validate inputs
        if not all([session_id, customer_id, api_key, user_message]):
            return jsonify({"error": "Missing required fields"}), 400

        # Validate customer credentials
        if not validate_customer(customer_id, api_key):
            return jsonify({"error": "Invalid Customer ID or API Key"}), 403

        # Get page context
        session_context = page_contexts.get(session_id)
        if not session_context or session_context['customerId'] != customer_id:
            return jsonify({"error": "Invalid session ID"}), 403

        # Format the context using the complete page data
        page_data = session_context['pageData']
        context = format_page_context(page_data)

        # Get AI response based on mode
        print("Calling backend -->", mode)
        if mode == "local":
            ai_message = get_response_from_ollama(user_message, context)
        elif mode == "remote":
            ai_message = get_response_from_remote(user_message, context)
        else:
            return jsonify({"error": "Invalid MODE configuration"}), 500

        # Check AI response
        if ai_message:
            formatted_response = {
                "response": ai_message,
                "format": "markdown",
            }
            print("AI Response:", formatted_response)
            return jsonify(formatted_response)
        else:
            return jsonify({"error": "Error communicating with the AI model"}), 500

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    mode = os.getenv("MODE", "remote")
    print("Service will use mode -> ", mode)
    port = int(os.getenv("PORT", 5000))
    print("Service will run on port -> ", port)
    if mode == 'remote':
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        app.run(debug=True)