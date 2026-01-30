"""
Test script for SPS AI API endpoints
Run this after starting the server with: python manage.py runserver 8001
"""

import requests
import json

BASE_URL = "http://localhost:8001/api"

# You'll need a valid JWT token from your sps_backend
# Replace this with your actual token
JWT_TOKEN = "your-jwt-token-here"

headers = {
    "Authorization": f"Bearer {JWT_TOKEN}",
    "Content-Type": "application/json"
}

def test_health():
    """Test health check endpoint"""
    print("\n🔍 Testing /health/...")
    try:
        response = requests.get(f"{BASE_URL.replace('/api', '')}/health/")
        print(f"✅ Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_conversations_list():
    """Test listing conversations"""
    print("\n🔍 Testing GET /api/conversations/...")
    try:
        response = requests.get(f"{BASE_URL}/conversations/", headers=headers)
        print(f"✅ Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_chat():
    """Test sending a chat message"""
    print("\n🔍 Testing POST /api/chat/...")
    payload = {
        "message": "Hello! Can you help me understand my pipeline model?",
        "include_rag": True
    }
    try:
        response = requests.post(f"{BASE_URL}/chat/", headers=headers, json=payload)
        print(f"✅ Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_documents_list():
    """Test listing documents"""
    print("\n🔍 Testing GET /api/documents/...")
    try:
        response = requests.get(f"{BASE_URL}/documents/", headers=headers)
        print(f"✅ Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("SPS AI API ENDPOINT TESTS")
    print("=" * 60)
    print("\n⚠️  Make sure:")
    print("1. Server is running: python manage.py runserver 8001")
    print("2. You have a valid JWT token from sps_backend")
    print("3. Update JWT_TOKEN variable in this script")
    print("=" * 60)
    
    # Run tests
    test_health()
    test_conversations_list()
    test_documents_list()
    test_chat()
    
    print("\n" + "=" * 60)
    print("✅ TESTS COMPLETE")
    print("=" * 60)
