import sys
import os
import random
import numpy as np
from dotenv import load_dotenv

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Load environment variables
load_dotenv()

# Set credentials from environment variable
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    raise EnvironmentError("Please set GOOGLE_APPLICATION_CREDENTIALS environment variable")

# Import Firestore and model functions
from api.config.firebase_config import initialize_firebase, db
from api.models.finalmodel import compute_match_similarity
from tests.results.test_embeddings import generate_and_save_data

# Initialize Firestore
db = initialize_firebase()

def test_find_best_matches(num_matches=3):
    """
    Test the matching algorithm by finding best matches for a random user.
    
    Args:
        num_matches (int): Number of top matches to return
        
    Returns:
        dict: Results containing user info and top brand matches
    """
    # Get all the data using the existing function from test_embeddings
    print("Generating embeddings and loading data...")
    (user_descriptions, brand_descriptions,
     user_values_embeddings, user_details_embeddings,
     brand_values_embeddings, brand_details_embeddings,
     user_min_compensations, brand_max_budgets) = generate_and_save_data()
    
    # Pick a random user
    user_id = random.choice(list(user_values_embeddings.keys()))
    user_desc = next(u for u in user_descriptions if u['user_id'] == user_id)
    user_idx = [u['user_id'] for u in user_descriptions].index(user_id)
    
    print(f"\nSelected User ID: {user_id}")
    
    # Compute similarity with all brands
    print("\nComputing similarities with brands...")
    brand_similarities = []
    for brand_id in brand_values_embeddings.keys():
        brand_idx = [b['brand_id'] for b in brand_descriptions].index(brand_id)
        
        values_similarity, details_similarity, price_compatibility, overall_similarity = compute_match_similarity(
            user_values_embeddings[user_id],
            user_details_embeddings[user_id],
            brand_values_embeddings[brand_id],
            brand_details_embeddings[brand_id],
            user_min_compensations[user_idx],
            brand_max_budgets[brand_idx]
        )
        
        brand_desc = next(b for b in brand_descriptions if b['brand_id'] == brand_id)
        brand_similarities.append({
            'brand_id': brand_id,
            'values_similarity': values_similarity,
            'details_similarity': details_similarity,
            'overall_similarity': overall_similarity,
            'price_compatibility': price_compatibility,
            'brand_values': brand_desc['values_description'],
            'brand_details': brand_desc['details_description'],
            'brand_max_budget': brand_max_budgets[brand_idx]
        })
    
    # Sort brands by overall similarity and get top matches
    brand_similarities.sort(key=lambda x: x['overall_similarity'], reverse=True)
    top_matches = brand_similarities[:num_matches]
    
    return {
        'user_id': user_id,
        'user_values': user_desc['values_description'],
        'user_details': user_desc['details_description'],
        'user_min_compensation': user_min_compensations[user_idx],
        'top_matches': top_matches
    }

if __name__ == "__main__":
    print("Starting best matches test...")
    results = test_find_best_matches()
    
    # Create output string
    output = []
    output.append("\nRandom User Profile:")
    output.append(f"User ID: {results['user_id']}")
    output.append(f"Values: {results['user_values']}")
    output.append(f"Details: {results['user_details']}")
    output.append(f"Minimum Compensation: ${results['user_min_compensation']}")
    
    output.append("\nTop 3 Brand Matches:")
    for i, match in enumerate(results['top_matches'], 1):
        output.append(f"\n{i}. Brand ID: {match['brand_id']}")
        output.append(f"   Overall Similarity Score: {match['overall_similarity']:.4f}")
        output.append(f"   Values Similarity: {match['values_similarity']:.4f}")
        output.append(f"   Details Similarity: {match['details_similarity']:.4f}")
        output.append(f"   Price Compatibility: {match['price_compatibility']:.4f}")
        output.append(f"   Brand Values: {match['brand_values']}")
        output.append(f"   Brand Details: {match['brand_details']}")
        output.append(f"   Brand Max Budget: ${match['brand_max_budget']}")
    
    # Print to console
    print('\n'.join(output))
    
    # Write to file
    output_file = os.path.join(os.path.dirname(__file__), 'match_results.txt')
    with open(output_file, 'w') as f:
        f.write('\n'.join(output))
    
    print(f"\nResults have been saved to: {output_file}")
