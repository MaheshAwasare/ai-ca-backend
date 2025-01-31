from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, get_jwt_identity, jwt_required
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import ollama
import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from datetime import datetime, timedelta
import bcrypt
import secrets
import string
import psycopg2
from psycopg2.extras import DictCursor
from contextlib import contextmanager
from cachetools import TTLCache
from functools import wraps
import time
import json

# Load environment variables
load_dotenv()
MODEL_NAME = "mistral:latest"
HF_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
app = Flask(__name__)
CORS(app)

# Rate Limit Configuration
RATE_LIMITS = {
    'DEFAULT': ["1000 per day", "50 per hour"],
    'AUTHENTICATION': {
        'REGISTER': "5000 per minute",
        'LOGIN': "5000 per minute"
    },
    'API': {
        'PAGE_CONTEXT': "5000 per hour",
        'SEND_MESSAGE': "5000 per minute"
    }
}


@app.route('/api/validate-token', methods=['GET'])
@jwt_required()
def validate_token():
    logger.info("Processing token validation request")
    start_time = time.time()
    try:
        user = get_user_from_token()
        if not user:
            logger.warning("Token validation failed: Invalid token")
            return jsonify({'error': 'Invalid token'}), 401

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                # Get API key for the customer
                cur.execute(
                    """
                    SELECT api_key 
                    FROM api_keys 
                    WHERE customer_id = %s AND is_active = true 
                    ORDER BY created_at DESC 
                    LIMIT 1
                    """,
                    (user['customer_id'],)
                )
                api_key_record = cur.fetchone()

                if not api_key_record:
                    logger.warning(f"No active API key found for customer {user['customer_id']}")
                    return jsonify({'error': 'No active API key found'}), 401

                set_user_context(conn, user['id'])
                logger.info(
                    f"Token validation successful for user_id: {user['id']}, time taken: {time.time() - start_time:.2f}s")
                return jsonify({
                    'valid': True,
                    'user_id': user['id'],
                    'username': user['username'],
                    'customer_id': user['customer_id'],
                    'api_key': api_key_record['api_key']
                }), 200
    except Exception as e:
        logger.error(f"Token validation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Configure logging
def setup_logger():
    logger = logging.getLogger('aica_api')
    logger.setLevel(logging.INFO)

    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Create file handler for logging to file
    file_handler = RotatingFileHandler(
        'logs/aica_api.log',
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    file_handler.setLevel(logging.INFO)

    # Create console handler for logging to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()

# Cache Configuration
CACHE_TTL = 120  # 2 minutes in seconds
user_cache = TTLCache(maxsize=100, ttl=CACHE_TTL)
customer_cache = TTLCache(maxsize=100, ttl=CACHE_TTL)


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


def cache_key(*args, **kwargs):
    """Generate a cache key from arguments."""
    return str(args) + str(sorted(kwargs.items()))


# Database Configuration
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'options': f"-c search_path={os.getenv('DB_SCHEMA')}"
}
required_db_vars = ['DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_SCHEMA']
missing_vars = [var for var in required_db_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    logger.info("Establishing database connection")
    start_time = time.time()
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
        logger.info(f"Database connection successful. Time taken: {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise
    finally:
        conn.close()
        logger.info("Database connection closed")


@contextmanager
def get_db_cursor(commit=False):
    """Context manager for database cursors."""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=DictCursor)
        try:
            yield cursor
            if commit:
                conn.commit()
                logger.info("Database transaction committed")
        finally:
            cursor.close()


def set_user_context(conn, user_id):
    """Set the user context for RLS."""
    logger.info(f"Setting user context for user_id: {user_id}")
    with conn.cursor() as cur:
        cur.execute("SELECT set_config('app.current_user_id', %s, true)", (str(user_id),))


def get_user_from_token():
    user_id = get_jwt_identity()
    #logger.info("User Id received during API call--", user_id)
    if not user_id:
        logger.warning("No user_id found in JWT token")
        return None

    cache_key = f"user_{user_id}"
    user = user_cache.get(cache_key)
    if user is not None:
        logger.info(f"User {user_id} found in cache")
        return user

    logger.info(f"User {user_id} not found in cache, querying database")

    with get_db_cursor() as cur:
        cur.execute(
            "SELECT id, username, customer_id FROM users WHERE id = %s",
            (user_id,)
        )
        user = cur.fetchone()
        logger.info(f"User {user} fetched")
        if user:
            user_cache[cache_key] = user
            logger.info(f"User {user_id} cached successfully")
        else:
            logger.warning(f"User {user_id} not found in database")
        return user


def validate_customer(customer_id, api_key):
    logger.info(f"Validating customer credentials for customer_id: {customer_id}")

    # Check cache first
    cache_key = f"customer_{customer_id}_{api_key}"
    print("cache_key  ", cache_key)
    is_valid = customer_cache.get(cache_key)
    if is_valid is not None:
        logger.info(f"Customer validation result found in cache for customer_id: {customer_id}")
        return is_valid

    logger.info(f"Customer validation not found in cache, querying database for customer_id: {customer_id}")
    # If not in cache, query database
    with get_db_cursor() as cur:
        cur.execute(
            "SELECT EXISTS(SELECT 1 FROM api_keys WHERE customer_id = %s AND api_key = %s AND is_active = true)",
            (customer_id, api_key)
        )
        is_valid = cur.fetchone()[0]
        customer_cache[cache_key] = is_valid
        logger.info(f"Customer validation result cached for customer_id: {customer_id}")
        return is_valid


# JWT Configuration
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-this')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
jwt = JWTManager(app)


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
    logger.warning(f"Rate limit exceeded for IP: {get_remote_address()}")
    return jsonify(error="Rate limit exceeded", retry_after=e.description), 429


# Constants
MODEL_NAME = "mistral:latest"
HF_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Data storage for chat context
page_contexts = {}
session_customer_map = {}


def generate_api_key(length=32):
    """Generate a secure API key."""
    api_key = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(length))
    logger.info("Generated new API key")
    return api_key


# Authentication endpoints
@app.route('/api/register', methods=['POST'])
@limiter.limit(RATE_LIMITS['AUTHENTICATION']['REGISTER'])
def register():
    logger.info("Processing registration request")
    start_time = time.time()
    try:
        data = request.json
        username = data.get('username')
        customer_id = data.get('customerId')

        logger.info(f"Registration attempt for username: {username}, customer_id: {customer_id}")

        if not all([username, data.get('password'), customer_id]):
            logger.warning("Registration failed: Missing required fields")
            return jsonify({'error': 'Missing required fields'}), 400

        # Generate a new API key
        api_key = generate_api_key()

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                # Check if username exists
                cur.execute("SELECT EXISTS(SELECT 1 FROM users WHERE username = %s)", (username,))
                if cur.fetchone()[0]:
                    logger.warning(f"Registration failed: Username already exists: {username}")
                    return jsonify({'error': 'Username already exists'}), 409

                # Hash the password
                password_hash = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())

                # Insert new user
                cur.execute(
                    "INSERT INTO users (username, password_hash, customer_id) VALUES (%s, %s, %s) RETURNING id",
                    (username, password_hash.decode('utf-8'), customer_id)
                )
                user_id = cur.fetchone()['id']

                # Set user context for RLS
                set_user_context(conn, user_id)

                # Insert API key
                cur.execute(
                    "INSERT INTO api_keys (customer_id, api_key) VALUES (%s, %s)",
                    (customer_id, api_key)
                )
                conn.commit()

        logger.info(f"Registration successful for username: {username}, time taken: {time.time() - start_time:.2f}s")
        return jsonify({
            'message': 'User registered successfully',
            'api_key': api_key
        }), 201

    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/login', methods=['POST'])
@limiter.limit(RATE_LIMITS['AUTHENTICATION']['LOGIN'])
def login():
    logger.info("Processing login request")
    start_time = time.time()
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')  # Plain password from client

        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400

        logger.info(f"Login attempt for username: {username}")

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                # Get user from database
                cur.execute(
                    "SELECT * FROM users WHERE username = %s",
                    (username,)
                )
                user = cur.fetchone()

                if not user:
                    logger.warning(f"Login failed: User {username} not found")
                    return jsonify({'error': 'Invalid username or password'}), 401

                # Debug stored hash processing
                stored_hash = user['password_hash'].encode('utf-8')
                logger.info(f"Stored hash from DB: {user['password_hash']}")

                # Compare passwords using the stored hash's salt
                is_valid = bcrypt.checkpw(password.encode('utf-8'), stored_hash)
                logger.info(f"Password comparison result: {is_valid}")

                if not is_valid:
                    logger.warning(f"Login failed: Invalid password for user {username}")
                    return jsonify({'error': 'Invalid username or password'}), 401

                # Get API key for the customer
                cur.execute(
                    """
                    SELECT api_key 
                    FROM api_keys 
                    WHERE customer_id = %s AND is_active = true 
                    ORDER BY created_at DESC 
                    LIMIT 1
                    """,
                    (user['customer_id'],)
                )
                api_key_record = cur.fetchone()

                if not api_key_record:
                    logger.warning(f"No active API key found for customer {user['customer_id']}")
                    return jsonify({'error': 'No active API key found'}), 401

                # Create access token
                access_token = create_access_token(
                    identity=str(user['id']),
                    expires_delta=timedelta(days=1)
                )

                logger.info(
                    f"Login successful for username: {username}, time taken: {time.time() - start_time:.2f}s, access token: {access_token}")
                return jsonify({
                    'access_token': access_token,
                    'customer_id': user['customer_id'],
                    'api_key': api_key_record['api_key']
                }), 200

    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/page-context', methods=['POST'])
@jwt_required()
@limiter.limit(RATE_LIMITS['API']['PAGE_CONTEXT'])
def page_context():
    logger.info("Processing page context request")
    start_time = time.time()
    try:
        user = get_user_from_token()
        if not user:
            logger.warning("Page context request failed: Invalid token")
            return jsonify({'error': 'Invalid token'}), 401

        data = request.json
        customer_id = data.get('customerId')
        page_url = data.get('pageUrl')

        logger.info(f"Page context request for customer_id: {customer_id}, page_url: {page_url}")

        if not all([customer_id, data.get('apiKey'), page_url, data.get('pageContent')]):
            logger.warning("Page context request failed: Missing required fields")
            return jsonify({'error': 'Missing required fields'}), 400

        # Validate customer credentials
        logger.info(f"VALIDATION FLOW--Customer Id {customer_id}. API KEY {data['apiKey']}")
        '''
        if not validate_customer(customer_id, data['apiKey']):
            logger.warning(f"Page context request failed: Invalid customer credentials for customer_id: {customer_id}")
            return jsonify({'error': 'Invalid customer credentials'}), 401
        '''
        with get_db_connection() as conn:
            set_user_context(conn, user['id'])
            # Store page context
            page_contexts[page_url] = data['pageContent']
            print(f"For url : {page_url},  Page context : {page_contexts[page_url]}")
            session_customer_map[page_url] = customer_id

            logger.info(
                f"Page context stored successfully for page_url: {page_url}, time taken: {time.time() - start_time:.2f}s")
            return jsonify({'message': 'Page context stored successfully'}), 200
    except Exception as e:
        logger.error(f"Page context error: {str(e)}")
        return jsonify({'error': str(e)}), 500


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


@app.route('/api/send-message', methods=['POST'])
@jwt_required()
@limiter.limit(RATE_LIMITS['API']['SEND_MESSAGE'])
def send_message():
    logger.info("Processing send message request")
    start_time = time.time()
    try:
        user = get_user_from_token()
        if not user:
            logger.warning("Send message request failed: Invalid token")
            return jsonify({'error': 'Invalid token'}), 401

        data = request.json
        customer_id = data.get('customerId')
        page_url = data.get('pageUrl')
        message = data.get('message')
        session_id = data.get("sessionId")
        customer_id = data.get("customerId")
        api_key = data.get("apiKey")
        user_message = data.get("message")
        logger.info(f"Message request for customer_id: {customer_id}, page_url: {page_url}")

        if not all([customer_id, data.get('apiKey'), page_url, message]):
            logger.warning("Send message request failed: Missing required fields")
            return jsonify({'error': 'Missing required fields'}), 400

        # Validate customer credentials
        print(f"Finding Validation Result for {customer_id} and {data['apiKey']}")
        validation_result = validate_customer(customer_id, data['apiKey'])
        print(f"Validation Result {validation_result}")
        if not validation_result:
            logger.warning(f"Send message request failed: Invalid customer credentials for customer_id: {customer_id}")
            return jsonify({'error': 'Invalid customer credentials'}), 401

        # Get page context
        print(f"Getting page contexts --- {page_contexts}")

        context = page_contexts.get(page_url)
        print(f"context  {context   }")

        if not context:
            logger.warning(f"Send message request failed: No context found for page_url: {page_url}")
            return jsonify({'error': 'No context found for this page'}), 404

        page_data = context
        print(f"page_data found {context}")
        context = format_page_context(page_data)
        print(f"context found   {context}")
        print("Calling backend -->", mode)
        if mode == "local":
            ai_message = get_response_from_ollama(user_message, context)
        elif mode == "remote":
            ai_message = get_response_from_remote(user_message, context)
        else:
            return jsonify({"error": "Invalid MODE configuration"}), 500

        # Check AI response
        if ai_message is None:
            ai_message = "Sorry, I couldn't generate a response."
        elif not isinstance(ai_message, str):
            try:
                ai_message = ai_message.get('response', str(ai_message))
            except Exception as e:
                ai_message = "An error occurred while processing the response."

        formatted_response = {
            "response": ai_message,
            "format": "markdown",
        }

        # When returning
        return jsonify({
            'response': formatted_response['response'],
            'context': context
        }), 200

    except Exception as e:
        print(f"Exception {e}")
        logger.error(f"Send message error: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    mode = os.getenv("MODE", "remote")
    port = int(os.getenv("PORT", 5000))
    logger.info(f"Service running in {mode} mode on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
