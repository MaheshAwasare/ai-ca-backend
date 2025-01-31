from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, get_jwt_identity, jwt_required
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import ollama
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from datetime import datetime, timedelta
import bcrypt
import secrets
import string

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# JWT Configuration
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-this')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
jwt = JWTManager(app)

# Rate Limit Configuration
RATE_LIMITS = {
    'DEFAULT': ["200 per day", "50 per hour"],
    'AUTHENTICATION': {
        'REGISTER': "5 per minute",
        'LOGIN': "10 per minute"
    },
    'API': {
        'PAGE_CONTEXT': "100 per hour",
        'SEND_MESSAGE': "50 per minute"
    }
}


def rate_limit_key_func():
    if request.headers.get('Authorization'):
        try:
            return get_jwt_identity() or get_remote_address()
        except:
            return get_remote_address()
    return get_remote_address()


storage_uri = os.getenv('REDIS_URL', 'memory://')
limiter = Limiter(
    app=app,
    key_func=rate_limit_key_func,
    default_limits=RATE_LIMITS['DEFAULT'],
    storage_uri=storage_uri
)


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify(error="Rate limit exceeded", retry_after=e.description), 429


# Constants
MODEL_NAME = "mistral:latest"
HF_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Data storage
page_contexts = {}
session_customer_map = {}
users = {}
api_keys = {}  # Store API keys with their associated customer IDs


def generate_api_key(length=32):
    """Generate a secure API key."""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def save_api_key(customer_id, api_key):
    """Save API key to customers.txt file."""
    try:
        with open("customers.txt", "a") as f:
            f.write(f"{customer_id},{api_key}\n")
        return True
    except Exception as e:
        print(f"Error saving API key: {e}")
        return False


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


def format_page_context(page_data):
    """Format page data into a comprehensive context string."""
    context_parts = []

    context_parts.append(f"Page URL: {page_data.get('url', 'N/A')}")
    context_parts.append(f"Page Title: {page_data.get('title', 'N/A')}")

    meta_tags = page_data.get('metaTags', [])
    if meta_tags:
        meta_desc = next((tag['content'] for tag in meta_tags if tag.get('name') == 'description'), None)
        if meta_desc:
            context_parts.append(f"Meta Description: {meta_desc}")

    headings = page_data.get('headings', [])
    if headings:
        context_parts.append("\nPage Structure:")
        for heading in headings:
            level = heading.get('level', '').replace('H', '')
            text = heading.get('text', '')
            context_parts.append(f"{'  ' * (int(level) - 1)}• {text}")

    return "\n".join(context_parts)


def get_response_from_ollama(user_message, context):
    """Get response from Ollama backend with context."""
    system_prompt = f"""You are a helpful assistant. Use the following context:
    {context}"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
    )
    return response["message"]["content"] if response and "message" in response else None


def get_response_from_remote(user_message, context):
    """Get response from Hugging Face Inference Client with context."""
    client = InferenceClient(HF_MODEL_NAME, token=HF_API_TOKEN)
    system_prompt = f"""You are a helpful assistant. Use the following context:
    {context}"""

    output = client.chat.completions.create(
        model=HF_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=1024,
    )
    return output.choices[0].message.content if output and output.choices else None


def validate_customer(customer_id, api_key):
    """Validate customer credentials."""
    valid_api_key = customers_data.get(customer_id)
    return valid_api_key == api_key


# Authentication endpoints
@app.route('/api/register', methods=['POST'])
@limiter.limit(RATE_LIMITS['AUTHENTICATION']['REGISTER'])
def register():
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        customer_id = data.get('customerId')

        if not all([username, password, customer_id]):
            return jsonify({'error': 'Missing required fields'}), 400

        if username in users:
            return jsonify({'error': 'Username already exists'}), 409

        # Generate a new API key
        api_key = generate_api_key()

        # Save the API key to the customers file
        if not save_api_key(customer_id, api_key):
            return jsonify({'error': 'Failed to save API key'}), 500

        # Update the in-memory customers data
        customers_data[customer_id] = api_key

        # Hash the password and store user data
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        users[username] = {
            'password': hashed_password,
            'customer_id': customer_id
        }

        return jsonify({
            'message': 'User registered successfully',
            'api_key': api_key  # Return the API key to the user
        }), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/login', methods=['POST'])
@limiter.limit(RATE_LIMITS['AUTHENTICATION']['LOGIN'])
def login():
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')

        if not all([username, password]):
            return jsonify({'error': 'Missing username or password'}), 400

        user = users.get(username)
        if not user or not bcrypt.checkpw(password.encode('utf-8'), user['password']):
            return jsonify({'error': 'Invalid username or password'}), 401

        access_token = create_access_token(identity=username)
        return jsonify({
            'access_token': access_token,
            'customer_id': user['customer_id']
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Protected API endpoints
@app.route('/api/page-context', methods=['POST'])
@jwt_required()
@limiter.limit(RATE_LIMITS['API']['PAGE_CONTEXT'])
def store_page_context():
    try:
        current_user = get_jwt_identity()
        data = request.json
        session_id = data.get('sessionId')
        page_data = data.get('pageData')

        if not all([session_id, page_data]):
            return jsonify({'error': 'Missing required fields'}), 400

        user = users.get(current_user)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        page_contexts[session_id] = {
            'pageData': page_data,
            'timestamp': datetime.now().isoformat(),
            'customerId': user['customer_id']
        }

        session_customer_map[session_id] = user['customer_id']

        return jsonify({'message': 'Page context stored successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/send-message', methods=['POST'])
@jwt_required()
@limiter.limit(RATE_LIMITS['API']['SEND_MESSAGE'])
def send_message():
    try:
        current_user = get_jwt_identity()
        data = request.json
        session_id = data.get('sessionId')
        user_message = data.get('message')

        if not all([session_id, user_message]):
            return jsonify({'error': 'Missing required fields'}), 400

        user = users.get(current_user)
        if not user:
            return jsonify({'error': 'User not found'}), 404

        session_context = page_contexts.get(session_id)
        if not session_context or session_context['customerId'] != user['customer_id']:
            return jsonify({'error': 'Invalid session ID'}), 403

        context = format_page_context(session_context['pageData'])
        mode = os.getenv("MODE", "remote")

        if mode == "local":
            ai_message = get_response_from_ollama(user_message, context)
        else:
            ai_message = get_response_from_remote(user_message, context)

        if not ai_message:
            return jsonify({'error': 'Error communicating with AI model'}), 500

        return jsonify({
            'response': ai_message,
            'format': 'markdown'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    mode = os.getenv("MODE", "remote")
    port = int(os.getenv("PORT", 5000))
    print(f"Service running in {mode} mode on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)