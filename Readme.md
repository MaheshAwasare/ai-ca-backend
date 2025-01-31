<!-- Decorative header -->
<div align="center">
  
# 🌟 AI Chat Widget Backend API with RAG 🤖
[![Demo](https://img.shields.io/badge/Demo-Live%20Preview-brightgreen?style=for-the-badge)](https://jaicalink.netlify.app/demo.html)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-blue?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)
[![JWT](https://img.shields.io/badge/JWT-Auth-red?style=for-the-badge&logo=json-web-tokens)](https://jwt.io/)

*A powerful backend API for the AI Chat Widget with JWT authentication and token validation*

---
</div>

## ✨ Features

- 🔐 **Secure Authentication**
  - JWT token-based authentication
  - Token validation endpoints
  - Secure password handling

- 🔑 **API Key Management**
  - Automatic API key retrieval
  - Active key validation
  - Customer-specific key management

- 👤 **User Context**
  - User identification
  - Customer association
  - Session management

## 🚀 Quick Start

1. **Set Environment Variables**
   ```env
   JWT_SECRET_KEY=your-super-secret-key-change-this-in-production
   MODE=remote
   PORT=5000
   HF_API_TOKEN=your-huggingface-token

   # Database Configuration
   DB_NAME=aica
   DB_USER=aica_user
   DB_PASSWORD=test
   DB_HOST=10-or-postgres.test.com
   DB_SCHEMA=aica_schema,public
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Server**
   ```bash
   python app.py
   ```

## 🔍 API Endpoints

### Token Validation
```http
GET /api/validate-token
```

**Headers:**
- `Authorization`: Bearer token
- `Content-Type`: application/json
- `X-API-Key`: API key

**Response:**
```json
{
  "valid": true,
  "user_id": "user_uuid",
  "username": "user_name",
  "customer_id": "customer_id",
  "api_key": "active_api_key"
}
```

## 🔧 Dependencies

```plaintext
annotated-types==0.7.0
anyio==4.8.0
bcrypt==4.2.1
blinker==1.9.0
cachetools==5.5.1
certifi==2024.12.14
charset-normalizer==3.4.1
click==8.1.8
colorama==0.4.6
Deprecated==1.2.18
distro==1.9.0
filelock==3.17.0
Flask==3.1.0
Flask-Cors==5.0.0
Flask-JWT-Extended==4.7.1
Flask-Limiter==3.10.1
fsspec==2024.12.0
greenlet==3.1.1
h11==0.14.0
httpcore==1.0.7
httpx==0.28.1
huggingface-hub==0.27.1
idna==3.10
itsdangerous==2.2.0
Jinja2==3.1.5
jiter==0.8.2
limits==4.0.1
markdown-it-py==3.0.0
MarkupSafe==3.0.2
mdurl==0.1.2
ollama==0.4.7
openai==1.60.0
ordered-set==4.1.0
packaging==24.2
psycopg2-binary==2.9.10
pydantic==2.10.5
pydantic_core==2.27.2
Pygments==2.19.1
PyJWT==2.10.1
python-dotenv==1.0.1
PyYAML==6.0.2
requests==2.32.3
rich==13.9.4
sniffio==1.3.1
SQLAlchemy==2.0.37
tqdm==4.67.1
typing_extensions==4.12.2
urllib3==2.3.0
Werkzeug==3.1.3
wrapt==1.17.2

```

## 🌐 Live Demo

Check out our live demo at [https://jaicalink.netlify.app/demo.html](https://jaicalink.netlify.app/demo.html)

## 🔒 Security Features

- JWT token authentication
- API key validation
- Database connection pooling
- User context management
- Error logging and monitoring

## 💡 Best Practices

- Secure credential handling
- Token-based authentication
- Database connection management
- Error handling and logging
- API key rotation support

<div align="center">

---

Made with ❤️ by the AI Chat Widget Team

[Live Demo](https://jaicalink.netlify.app/demo.html) • [Report Bug](https://github.com/yourusername/your-repo/issues) • [Request Feature](https://github.com/yourusername/your-repo/issues)

</div>
