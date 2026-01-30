# SPS AI Service

Django-based AI/Chat service for SPS Assistant.

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with API keys

# Database
python manage.py migrate

# Run (port 8001)
python manage.py runserver 8001
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/api/chat/` | Send message |
| `/api/conversations/` | List conversations |
| `/api/suggestions/` | Model analysis |
| `/health/` | Health check |

## Configuration

Set `AI_LLM_PROVIDER` to `anthropic` or `openai` in `.env`

## Related Services

- **sps_backend** (port 8000)
- **sps_frontend** (port 5173)
