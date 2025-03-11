"""
Brand-User Recommendation Engine Core Model

This module implements the core recommendation system using BERT-based embeddings
and similarity computations. It provides functionality for:

1. Embedding Generation:
   - Compute embeddings for user and brand profiles
   - Generate semantic representations of values and details
   - Update embeddings based on interactions

2. Similarity Matching:
   - Multi-dimensional similarity computation
   - Price compatibility analysis
   - Weighted scoring system

3. Database Integration:
   - Firestore transaction management
   - Retry mechanisms for resilience
   - Atomic updates for consistency
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from google.cloud import firestore
from google.api_core.exceptions import DeadlineExceeded, ServiceUnavailable
from api.config.firebase_config import db
from datetime import datetime
import logging
from google.api_core.retry import Retry

logger = logging.getLogger('api.models.finalmodel')
logger.propagate = True

def compute_embedding(text, model):
    """
    Compute sentence embedding using the provided BERT model.
    
    Args:
        text (str): Input text to encode
        model (SentenceTransformer): Pre-trained sentence transformer model
        
    Returns:
        numpy.ndarray: Embedding vector
    """
    return model.encode(text)

def create_user_description(user_data):
    """Generate descriptive strings for a user's values and details."""
    # Extract fields with defaults
    type_of_user = user_data.get("userType", "user")
    category = user_data.get("category", "their field")
    values = user_data.get("valuesInterests", [])
    location = user_data.get("location", None)
    affiliation = user_data.get("affiliation", None)

    # Create values description
    if values:
        values_description = (
            f"Primary interest is {values[0]}" if len(values) == 1
            else f"Primary interests are {', '.join(values[:-1])} and {values[-1]}"
        )
    else:
        values_description = "No specific interests listed"

    # Create details description
    details_description = f"A {type_of_user} in {category}"
    if location and pd.notna(location):
        details_description += f". Based in {location}"
    if affiliation and pd.notna(affiliation) and affiliation not in ['nan', None]:
        details_description += f". Affiliated with {affiliation}"
    details_description += "."

    return values_description, details_description

def create_brand_description(brand_data):
    """Generate descriptive strings for a brand's values and details."""
    # Extract fields with defaults
    industry = brand_data.get("industry", "various industries")
    company_values = brand_data.get("companyValues", [])
    company_size = brand_data.get("companySize", None)
    country = brand_data.get("country", None)
    state = brand_data.get("state", None)

    # Create values description
    if company_values:
        values_description = (
            f"Primary focus is {company_values[0]}" if len(company_values) == 1
            else f"Primary focuses are {', '.join(company_values[:-1])} and {company_values[-1]}"
        )
    else:
        values_description = "No specific focuses listed"

    # Create details description
    details_description = f"Operates in the {industry} industry"
    if company_size and pd.notna(company_size):
        details_description += f". A {company_size} company"
    if country and pd.notna(country) and state and pd.notna(state):
        details_description += f". Located in {state}, {country}"
    elif country and pd.notna(country):
        details_description += f". Located in {country}"
    details_description += "."

    return values_description, details_description

def update_embeddings(
    current_values_embedding,
    current_details_embedding,
    new_values_embedding,
    new_details_embedding,
    alpha=0.7
):
    """Update embeddings using exponentially weighted average."""
    # Compute weighted average
    updated_values_embedding = (
        alpha * current_values_embedding + 
        (1 - alpha) * new_values_embedding
    )
    updated_details_embedding = (
        alpha * current_details_embedding + 
        (1 - alpha) * new_details_embedding
    )

    # Normalize to unit length
    updated_values_embedding /= np.linalg.norm(updated_values_embedding)
    updated_details_embedding /= np.linalg.norm(updated_details_embedding)
    
    return updated_values_embedding, updated_details_embedding

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def compute_price_compatibility(
    athlete_min, 
    brand_max, 
    lower_buffer=0.3, 
    upper_buffer=0.5
):
    """Calculate price compatibility score between athlete and brand."""
    negotiation_threshold = brand_max * (1 - lower_buffer)
    max_threshold = brand_max * (1 + upper_buffer)

    if athlete_min <= negotiation_threshold:
        return 1.0
    elif athlete_min <= max_threshold:
        return 1 - (
            (athlete_min - negotiation_threshold) / 
            (max_threshold - negotiation_threshold)
        )
    return 0.0

def compute_match_similarity(
    user_values_embedding, 
    user_details_embedding, 
    brand_values_embedding, 
    brand_details_embedding,
    user_min_compensation,
    brand_max_budget,
    values_weight=0.4,
    details_weight=0.3,
    price_weight=0.3
):
    """Compute similarity scores between user and brand embeddings."""
    # Validate weights
    if not abs((values_weight + details_weight + price_weight) - 1.0) < 1e-9:
        raise ValueError("Weights must sum to 1.0")

    # Compute similarities
    values_similarity = cosine_similarity(user_values_embedding, brand_values_embedding)
    details_similarity = cosine_similarity(user_details_embedding, brand_details_embedding)
    price_compatibility = compute_price_compatibility(user_min_compensation, brand_max_budget)
    
    # Compute weighted overall similarity
    overall_similarity = (
        values_weight * values_similarity + 
        details_weight * details_similarity + 
        price_weight * price_compatibility
    )
    
    return values_similarity, details_similarity, price_compatibility, overall_similarity

# Define a custom predicate for retryable exceptions
def is_retryable_exception(exc):
    return isinstance(exc, (DeadlineExceeded, ServiceUnavailable))

# Use the Retry class with the custom predicate
retry_decorator = Retry(predicate=is_retryable_exception, deadline=30)

@retry_decorator
def fetch_document(doc_ref, timeout=5):
    """Fetch a Firestore document with retries and consistent error handling."""
    try:
        doc_snapshot = doc_ref.get(timeout=timeout)
        if not doc_snapshot.exists:
            raise ValueError(f"Document not found for reference: {doc_ref.path}")
        return doc_snapshot
    except DeadlineExceeded:
        logger.error(f"Timeout fetching document: {doc_ref.path}")
        raise TimeoutError(f"Firestore request timed out for document: {doc_ref.path}")
    except Exception as e:
        logger.error(f"Failed to fetch document {doc_ref.path}: {str(e)}")
        raise

def compute_initial_user_embedding(user_id, db, model):
    """
    Compute initial embeddings for a user.
    
    Args:
        user_id (str): User ID to compute embeddings for
        db (firestore.Client): Firestore database client
        model (SentenceTransformer): Pre-trained sentence transformer model
        
    Returns:
        dict: Generated embeddings data
        
    Raises:
        ValueError: If user not found
        TimeoutError: If Firestore request times out
        Exception: For other computation errors
    """
    logger.info(f"Computing embeddings for user: {user_id}")
    try:
        # Fetch user data
        user_ref = db.collection('Users').document(user_id)
        user_doc = fetch_document(user_ref)
        user_data = user_doc.to_dict()

        # Generate descriptions and compute embeddings
        values_description, details_description = create_user_description(user_data)
        values_embedding = compute_embedding(values_description, model)
        details_embedding = compute_embedding(details_description, model)
        
        # Store embeddings
        embedding_data = {
            'values_embedding': values_embedding.tolist(),
            'details_embedding': details_embedding.tolist(),
            'last_updated': firestore.SERVER_TIMESTAMP
        }

        user_ref.update({'embeddings': embedding_data})
        logger.info(f"Embeddings stored for user: {user_id}")

        return {
            'values_embedding': embedding_data['values_embedding'],
            'details_embedding': embedding_data['details_embedding']
        }

    except TimeoutError as e:
        logger.error(f"Timeout computing embeddings for user {user_id}: {e}")
        return {"error": "Firestore request timed out"}, 504
    except ValueError as e:
        logger.error(f"User not found: {user_id}")
        return {"error": f"User document not found: {user_id}"}, 404
    except Exception as e:
        logger.error(f"Failed to compute embeddings for user {user_id}: {str(e)}")
        raise

def compute_initial_brand_embedding(brand_id, db, model):
    """Compute initial embeddings for a brand"""
    logger.info(f"Computing embeddings for brand: {brand_id}")
    try:
        # Fetch brand data
        brand_ref = db.collection('Brands').document(brand_id)
        brand_doc = fetch_document(brand_ref)
        brand_data = brand_doc.to_dict()

        # Generate descriptions and compute embeddings
        values_description, details_description = create_brand_description(brand_data)
        values_embedding = compute_embedding(values_description, model)
        details_embedding = compute_embedding(details_description, model)
        
        # Store embeddings
        embedding_data = {
            'values_embedding': values_embedding.tolist(),
            'details_embedding': details_embedding.tolist(),
            'last_updated': firestore.SERVER_TIMESTAMP
        }
        
        brand_ref.update({'embeddings': embedding_data})
        logger.info(f"Embeddings stored for brand: {brand_id}")
        
        return {
            'values_embedding': embedding_data['values_embedding'],
            'details_embedding': embedding_data['details_embedding']
        }
        
    except TimeoutError as e:
        logger.error(f"Timeout computing embeddings for brand {brand_id}: {e}")
        return {"error": "Firestore request timed out"}, 504
    except ValueError as e:
        logger.error(f"Brand not found: {brand_id}")
        return {"error": f"Brand document not found: {brand_id}"}, 404
    except Exception as e:
        logger.error(f"Failed to compute embeddings for brand {brand_id}: {str(e)}")
        raise

def get_top_brand_matches(user_id, db, n_matches=10):
    """Get top brand matches for a user"""
    logger.info(f"Finding top {n_matches} brand matches for user: {user_id}")
    try:
        user_ref = db.collection('Users').document(user_id)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            logger.error(f"User not found: {user_id}")
            raise ValueError(f"No user found with ID: {user_id}")
            
        user_data = user_doc.to_dict()
        
        if 'embeddings' not in user_data:
            logger.error(f"No embeddings found for user: {user_id}")
            raise ValueError("User embeddings not found")
            
        # Compute matches
        matches = []
        brands = db.collection('Brands').get()
        
        for brand in brands:
            brand_data = brand.to_dict()
            if 'embeddings' not in brand_data:
                continue
                
            # Compute similarity scores and append match
            values_sim, details_sim, price_comp, overall_sim = compute_match_similarity(
                np.array(user_data['embeddings']['values_embedding']),
                np.array(user_data['embeddings']['details_embedding']),
                np.array(brand_data['embeddings']['values_embedding']),
                np.array(brand_data['embeddings']['details_embedding']),
                user_data.get('minCompensation', 0),
                brand_data.get('maxBudget', float('inf'))
            )
            
            matches.append({
                'brand_id': brand.id,
                'brand_name': brand_data.get('name', 'Unknown'),
                'similarity_scores': {
                    'values_similarity': float(values_sim),
                    'details_similarity': float(details_sim),
                    'price_compatibility': float(price_comp),
                    'overall_similarity': float(overall_sim)
                }
            })
        
        # Sort and return top matches
        matches.sort(key=lambda x: x['similarity_scores']['overall_similarity'], reverse=True)
        top_matches = matches[:n_matches]
        
        logger.info(f"Found {len(top_matches)} matches for user: {user_id}")
        return {
            "success": True,
            "user_id": user_id,
            "number_of_matches": len(top_matches),
            "matches": top_matches
        }
        
    except Exception as e:
        logger.error(f"Error finding matches for user {user_id}: {str(e)}")
        raise

def get_top_user_matches(brand_id, db, n_matches=10):
    """Get the top N user matches for a given brand."""
    logger.info(f"Finding top {n_matches} user matches for brand: {brand_id}")
    try:
        # Get brand data from Firestore
        brand_ref = db.collection('Brands').document(brand_id)
        brand_doc = brand_ref.get()
        
        if not brand_doc.exists:
            logger.error(f"Brand not found: {brand_id}")
            raise ValueError(f"No brand found with ID: {brand_id}")
            
        brand_data = brand_doc.to_dict()
        
        if 'embeddings' not in brand_data:
            logger.error(f"No embeddings found for brand: {brand_id}")
            raise ValueError("Brand embeddings not found")
            
        brand_embeddings = brand_data['embeddings']
        brand_values_embedding = np.array(brand_embeddings['values_embedding'])
        brand_details_embedding = np.array(brand_embeddings['details_embedding'])
        
        # Get all users and compute matches
        users = db.collection('Users').get()
        matches = []
        
        for user in users:
            user_data = user.to_dict()
            if 'embeddings' not in user_data:
                continue
                
            # Compute similarity scores
            values_sim, details_sim, price_comp, overall_sim = compute_match_similarity(
                np.array(user_data['embeddings']['values_embedding']),
                np.array(user_data['embeddings']['details_embedding']),
                brand_values_embedding,
                brand_details_embedding,
                user_data.get('minCompensation', 0),
                brand_data.get('maxBudget', float('inf'))
            )
            
            matches.append({
                'user_id': user.id,
                'user_name': user_data.get('name', 'Unknown'),
                'similarity_scores': {
                    'values_similarity': float(values_sim),
                    'details_similarity': float(details_sim),
                    'price_compatibility': float(price_comp),
                    'overall_similarity': float(overall_sim)
                }
            })
        
        # Sort matches by overall similarity and get top N
        matches.sort(key=lambda x: x['similarity_scores']['overall_similarity'], reverse=True)
        top_matches = matches[:n_matches]
        
        logger.info(f"Found {len(top_matches)} matches for brand: {brand_id}")
        return {
            "success": True,
            "brand_id": brand_id,
            "number_of_matches": len(top_matches),
            "matches": top_matches
        }
        
    except Exception as e:
        logger.error(f"Error finding matches for brand {brand_id}: {str(e)}")
        raise

def update_user_embeddings(user_id, brand_id, db, alpha=0.7):
    """Update a user's embeddings based on interaction with a brand using transactions."""
    logger.info(f"Updating user embeddings - user: {user_id}, brand: {brand_id}")
    try:
        user_ref = db.collection('Users').document(user_id)
        brand_ref = db.collection('Brands').document(brand_id)

        @firestore.transactional
        def update_in_transaction(transaction):
            # Get documents in transaction
            user_doc = user_ref.get(transaction=transaction)
            brand_doc = brand_ref.get(transaction=transaction)
            
            if not user_doc.exists:
                logger.error(f"User not found: {user_id}")
                raise ValueError(f"No user found with ID: {user_id}")
            if not brand_doc.exists:
                logger.error(f"Brand not found: {brand_id}")
                raise ValueError(f"No brand found with ID: {brand_id}")
                
            user_data = user_doc.to_dict()
            brand_data = brand_doc.to_dict()
            
            if 'embeddings' not in user_data or 'embeddings' not in brand_data:
                logger.error(f"Missing embeddings - user: {user_id}, brand: {brand_id}")
                raise ValueError("Required embeddings not found")
            
            # Update embeddings
            updated_values_embedding, updated_details_embedding = update_embeddings(
                np.array(user_data['embeddings']['values_embedding']),
                np.array(user_data['embeddings']['details_embedding']),
                np.array(brand_data['embeddings']['values_embedding']),
                np.array(brand_data['embeddings']['details_embedding']),
                alpha
            )
            
            # Prepare and perform update
            embedding_data = {
                'embeddings': {
                    'values_embedding': updated_values_embedding.tolist(),
                    'details_embedding': updated_details_embedding.tolist(),
                    'last_updated': firestore.SERVER_TIMESTAMP
                }
            }
            
            transaction.update(user_ref, embedding_data)
            logger.info(f"User embeddings updated - user: {user_id}")
            
            return {
                'success': True,
                'message': 'User embeddings updated successfully',
                'user_id': user_id,
                'brand_id': brand_id
            }

        # Execute transaction
        transaction = db.transaction()
        return update_in_transaction(transaction)
        
    except Exception as e:
        logger.error(f"Failed to update user embeddings - user: {user_id}, brand: {brand_id}, error: {str(e)}")
        raise

def update_brand_embeddings(brand_id, user_id, db, alpha=0.7):
    """Update a brand's embeddings based on interaction with a user using transactions."""
    logger.info(f"Updating brand embeddings - brand: {brand_id}, user: {user_id}")
    try:
        brand_ref = db.collection('Brands').document(brand_id)
        user_ref = db.collection('Users').document(user_id)

        @firestore.transactional
        def update_in_transaction(transaction):
            # Get documents in transaction
            brand_doc = brand_ref.get(transaction=transaction)
            user_doc = user_ref.get(transaction=transaction)
            
            if not brand_doc.exists:
                logger.error(f"Brand not found: {brand_id}")
                raise ValueError(f"No brand found with ID: {brand_id}")
            if not user_doc.exists:
                logger.error(f"User not found: {user_id}")
                raise ValueError(f"No user found with ID: {user_id}")
                
            brand_data = brand_doc.to_dict()
            user_data = user_doc.to_dict()
            
            if 'embeddings' not in brand_data or 'embeddings' not in user_data:
                logger.error(f"Missing embeddings - brand: {brand_id}, user: {user_id}")
                raise ValueError("Required embeddings not found")
            
            # Update embeddings
            updated_values_embedding, updated_details_embedding = update_embeddings(
                np.array(brand_data['embeddings']['values_embedding']),
                np.array(brand_data['embeddings']['details_embedding']),
                np.array(user_data['embeddings']['values_embedding']),
                np.array(user_data['embeddings']['details_embedding']),
                alpha
            )
            
            # Prepare and perform update
            embedding_data = {
                'embeddings': {
                    'values_embedding': updated_values_embedding.tolist(),
                    'details_embedding': updated_details_embedding.tolist(),
                    'last_updated': firestore.SERVER_TIMESTAMP
                }
            }
            
            transaction.update(brand_ref, embedding_data)
            logger.info(f"Brand embeddings updated - brand: {brand_id}")
            
            return {
                'success': True,
                'message': 'Brand embeddings updated successfully',
                'brand_id': brand_id,
                'user_id': user_id
            }

        # Execute transaction
        transaction = db.transaction()
        return update_in_transaction(transaction)
        
    except Exception as e:
        logger.error(f"Failed to update brand embeddings - brand: {brand_id}, user: {user_id}, error: {str(e)}")
        raise

