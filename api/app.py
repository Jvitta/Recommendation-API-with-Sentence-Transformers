"""
Brand-User Recommendation System API

This Flask application provides a RESTful API for the recommendation system.
It handles user and brand profile management, embedding generation, and match computation.

Key Features:
1. Authentication using Firebase Admin SDK
2. Comprehensive error handling and logging
3. Cloud-ready configuration
4. Production-grade security practices

The API is designed to be scalable and maintainable, with proper separation of concerns
and robust error handling throughout.
"""

from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, auth
from functools import wraps
from api.config.firebase_config import db 
from api.models.finalmodel import (
    compute_initial_user_embedding,
    compute_initial_brand_embedding,
    get_top_user_matches,
    get_top_brand_matches,
    update_user_embeddings,
    update_brand_embeddings,
)
import os
import logging
import sys
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler
import google.cloud.logging_v2.resource
import traceback
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Pre-load the model for efficiency
model = SentenceTransformer('all-MiniLM-L6-v2')

def setup_logging():
    """
    Initialize logging configuration for the application.
    
    Configures logging differently for Cloud Run vs local development:
    - Cloud Run: Uses Cloud Logging with stderr fallback
    - Local: Uses stderr logging
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger('api')
    logger.setLevel(logging.INFO)
    
    if os.getenv('K_SERVICE'):  # Running in Cloud Run
        try:
            client = google.cloud.logging.Client()
            handler = CloudLoggingHandler(client)
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            
            logger.handlers = []
            logger.addHandler(handler)
            
            # Add stderr handler for container logs
            stderr_handler = logging.StreamHandler(sys.stderr)
            stderr_handler.setFormatter(formatter)
            logger.addHandler(stderr_handler)
            
            logger.info("Cloud Logging initialized")
            
        except Exception as e:
            print(f"Failed to initialize Cloud Logging: {e}", file=sys.stderr)
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            logger.handlers = [handler]
    else:
        # Local development logging
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.handlers = [handler]
    
    return logger

# Initialize logging and Flask app
logger = setup_logging()
app = Flask(__name__)
app.logger.handlers = logger.handlers
app.logger.setLevel(logging.INFO)

# Request logging middleware
@app.before_request
def log_request_info():
    """Log incoming request details except health checks."""
    if request.path != '/health':
        logger.info(f"Request: {request.method} {request.path}")

@app.after_request
def log_response_info(response):
    """Log response status for non-200 responses except health checks."""
    if request.path != '/health' and response.status_code >= 400:
        logger.error(f"Response: {response.status}")
    return response

def authenticate(f):
    """
    Authentication decorator using Firebase Admin SDK.
    
    Verifies the Bearer token in the Authorization header and attaches
    the decoded user information to the request.
    
    Returns:
        function: Decorated route handler
        
    Raises:
        401: If token is missing, invalid, or expired
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        logger.info("Starting authentication")
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            logger.error("Unauthorized request: Missing or invalid token")
            return jsonify({"error": "Unauthorized: Missing or invalid token"}), 401
        
        id_token = auth_header.split(" ")[1]
        try:
            decoded_token = auth.verify_id_token(id_token)
            request.user = decoded_token
            logger.info(f"Successfully authenticated user: {decoded_token['uid']}")
            return f(*args, **kwargs)
        except auth.ExpiredIdTokenError:
            logger.error("Token validation failed: Token has expired")
            return jsonify({"error": "Unauthorized: Token expired"}), 401
        except Exception as e:
            logger.error(f"Token validation failed: {str(e)}")
            return jsonify({"error": "Unauthorized: Invalid or expired token"}), 401
    return decorated_function

# Error handling for unknown routes
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    """Handle unknown routes with a 404 response."""
    logger.warning(f"Received request for unknown path: {path}")
    return jsonify({"error": f"Path not found: {path}"}), 404

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring system status.
    
    Returns:
        dict: Health status and timestamp
    """
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/set-user-embedding', methods=['POST'])
@authenticate
def set_user_embedding():
    data = request.get_json()
    user_id = data.get('user_id')
    
    if not user_id:
        logger.error("Missing user_id in request")
        return jsonify({"error": "user_id is required"}), 400
        
    try:
        embedding_data = compute_initial_user_embedding(user_id, db, model)
        logger.info(f"Successfully set embeddings for user: {user_id}")
        return jsonify({
            "success": True,
            "user_id": user_id,
            "embedding_data": embedding_data
        }), 200
    except ValueError as ve:
        logger.error(f"ValueError in set_user_embedding: {ve}")
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        logger.error(f"Error in set_user_embedding: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/set-brand-embedding', methods=['POST'])
@authenticate
def set_brand_embedding():
    logger.debug("Set brand embedding endpoint hit")
    data = request.get_json()
    logger.debug(f"Received data: {data}")
    brand_id = data.get('brand_id')
    
    if not brand_id:
        logger.error("Missing brand_id in request")
        return jsonify({"error": "brand_id is required"}), 400
        
    try:
        embedding_data = compute_initial_brand_embedding(brand_id, db, model)
        return jsonify({
            "success": True,
            "brand_id": brand_id,
            "embedding_data": embedding_data
        }), 200
    except ValueError as ve:
        logger.error(f"ValueError in set_brand_embedding: {ve}")
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        logger.exception("Unexpected error in set_brand_embedding")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/get-brand-matches', methods=['GET'])
@authenticate
def get_brand_matches():
    user_id = request.args.get('user_id')
    n_matches = request.args.get('n', default=10, type=int)
    
    if not user_id:
        logger.error("Missing user_id in request")
        return jsonify({"error": "user_id is required"}), 400
        
    try:
        result = get_top_brand_matches(user_id, db, n_matches)
        logger.info(f"Found {result['number_of_matches']} matches for user: {user_id}")
        return jsonify(result), 200
    except ValueError as ve:
        logger.error(f"ValueError in get_brand_matches: {ve}")
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        logger.error(f"Error in get_brand_matches: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/get-user-matches', methods=['GET'])
@authenticate
def get_user_matches():
    brand_id = request.args.get('brand_id')
    n_matches = request.args.get('n', default=10, type=int)
    
    if not brand_id:
        logger.error("Missing brand_id in request")
        return jsonify({"error": "brand_id is required"}), 400
        
    try:
        result = get_top_user_matches(brand_id, db, n_matches)
        logger.info(f"Found {result['number_of_matches']} matches for brand: {brand_id}")
        return jsonify(result), 200
    except ValueError as ve:
        logger.error(f"ValueError in get_user_matches: {ve}")
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        logger.error(f"Error in get_user_matches: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/update-user-embedding', methods=['POST'])
@authenticate
def update_user_embedding():
    data = request.get_json()
    user_id = data.get('user_id')
    brand_id = data.get('brand_id')
    alpha = data.get('alpha', 0.7)
    
    if not user_id or not brand_id:
        logger.error("Missing user_id or brand_id in request")
        return jsonify({"error": "Both user_id and brand_id are required"}), 400
        
    try:
        update_info = update_user_embeddings(user_id, brand_id, db, alpha)
        logger.info(f"Successfully updated embeddings for user {user_id} with brand {brand_id}")
        return jsonify({
            "success": True,
            "user_id": update_info["user_id"],
            "brand_id": update_info["brand_id"],
            "message": update_info.get("message", "Update successful")
        }), 200
    except ValueError as ve:
        logger.error(f"ValueError in update_user_embedding: {ve}")
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        logger.error(f"Error updating user embedding - user: {user_id}, brand: {brand_id}, error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/update-brand-embedding', methods=['POST'])
@authenticate
def update_brand_embedding():
    data = request.get_json()
    brand_id = data.get('brand_id')
    user_id = data.get('user_id')
    alpha = data.get('alpha', 0.7)
    
    if not brand_id or not user_id:
        logger.error("Missing brand_id or user_id in request")
        return jsonify({"error": "Both brand_id and user_id are required"}), 400
        
    try:
        update_info = update_brand_embeddings(brand_id, user_id, db, alpha)
        logger.info(f"Successfully updated embeddings for brand {brand_id} with user {user_id}")
        return jsonify({
            "success": True,
            "brand_id": brand_id,
            "user_id": user_id,
            "message": update_info.get("message", "Update successful")
        }), 200
    except ValueError as ve:
        logger.error(f"ValueError in update_brand_embedding: {ve}")
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        logger.error(f"Error updating brand embedding - brand: {brand_id}, user: {user_id}, error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)

