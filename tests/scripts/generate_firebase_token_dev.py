"""
Generate Firebase authentication tokens for development and testing.

This script:
1. Initializes Firebase Admin SDK
2. Generates a custom token
3. Exchanges it for an ID token
4. Saves both tokens to a file

Environment variables required:
- FIREBASE_WEB_API_KEY: Firebase Web API key from Firebase Console
- TEST_USER_ID: Firebase user ID for testing
"""

import firebase_admin
from firebase_admin import credentials, auth
import json
import time
import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

def get_credentials_path():
    """Get the path to Firebase credentials file."""
    return os.path.join(
        os.path.expanduser("~"),
        "OneDrive",
        "Documents",
        ".google",
        "credentials",
        "credentials_dev.json"
    )

def generate_tokens():
    """Generate and exchange Firebase tokens."""
    # Get required environment variables
    web_api_key = os.getenv('FIREBASE_WEB_API_KEY')
    user_id = os.getenv('TEST_USER_ID')

    if not web_api_key or not user_id:
        raise ValueError(
            "Missing required environment variables. Please set:\n"
            "- FIREBASE_WEB_API_KEY\n"
            "- TEST_USER_ID"
        )

    # Initialize Firebase Admin SDK
    cred = credentials.Certificate(get_credentials_path())
    try:
        firebase_admin.initialize_app(cred)
    except ValueError:
        # App already initialized
        pass

    # Generate Firebase custom token
    custom_token = auth.create_custom_token(user_id)

    # Exchange custom token for ID token
    exchange_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken?key={web_api_key}"
    exchange_payload = {
        "token": custom_token.decode('utf-8'),
        "returnSecureToken": True
    }

    response = requests.post(exchange_url, json=exchange_payload)
    response.raise_for_status()  # Raise exception for bad status codes
    id_token = response.json()['idToken']

    return custom_token, id_token

def save_tokens(custom_token, id_token):
    """Save tokens to file with expiry time."""
    expiry_time = int(time.time()) + 3600  # 1 hour from now
    token_data = {
        "custom_token": custom_token.decode('utf-8'),
        "id_token": id_token,
        "expires_at": expiry_time
    }

    # Save to the project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    token_file_path = os.path.join(project_root, "firebase_token.json")

    with open(token_file_path, "w") as f:
        json.dump(token_data, f)

    return token_file_path

def main():
    """Main execution function."""
    try:
        custom_token, id_token = generate_tokens()
        token_file_path = save_tokens(custom_token, id_token)
        
        print(f"Tokens saved to {token_file_path}")
        print("Token generation successful!")
        
    except Exception as e:
        print(f"Error generating tokens: {str(e)}")
        raise

if __name__ == "__main__":
    main()
