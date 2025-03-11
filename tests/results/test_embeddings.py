import sys
import os
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm import tqdm

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Load environment variables
load_dotenv()

# Check for credentials
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    raise EnvironmentError("Please set GOOGLE_APPLICATION_CREDENTIALS environment variable")

# Import Firestore and model functions
from api.config.firebase_config import initialize_firebase, db
from api.models.finalmodel import (
    create_user_description,
    create_brand_description,
    compute_embedding,
    get_model,
    compute_match_similarity,
    compute_initial_user_embedding,
    compute_initial_brand_embedding
)

# Initialize Firestore with production project
db = initialize_firebase()

def test_generate_and_save_data():
    """
    Test embedding generation and storage for users and brands.
    
    Returns:
        tuple: Contains user and brand descriptions, embeddings, and compensation data
    """
    # Initialize dictionaries to store embeddings
    user_values_embeddings = {}
    user_details_embeddings = {}
    brand_values_embeddings = {}
    brand_details_embeddings = {}

    # Fetch data from Firestore
    users = list(db.collection('Users').get())
    brands = list(db.collection('Brands').get())

    # Process in batches
    BATCH_SIZE = 10

    def process_batch(items, is_user=True):
        descriptions = []
        values_embeddings = {}
        details_embeddings = {}
        
        for item in items:
            item_id = item.id
            item_data = item.to_dict()
            
            try:
                # Create descriptions and compute embeddings
                if is_user:
                    values_desc, details_desc = create_user_description(item_data)
                    # Get embeddings directly from document since we're just testing
                    if 'embeddings' in item_data:
                        embedding_data = item_data['embeddings']
                    else:
                        embedding_data = compute_initial_user_embedding(item_id, db)
                else:
                    values_desc, details_desc = create_brand_description(item_data)
                    if 'embeddings' in item_data:
                        embedding_data = item_data['embeddings']
                    else:
                        embedding_data = compute_initial_brand_embedding(item_id, db)
                
                # Store embeddings
                values_embeddings[item_id] = np.array(embedding_data['values_embedding'])
                details_embeddings[item_id] = np.array(embedding_data['details_embedding'])
                
                descriptions.append({
                    f"{'user' if is_user else 'brand'}_id": item_id,
                    'values_description': values_desc,
                    'details_description': details_desc
                })
                
            except Exception as e:
                print(f"Error processing {'user' if is_user else 'brand'} {item_id}: {str(e)}")
                continue
                
        return descriptions, values_embeddings, details_embeddings

    # Process users in batches with progress bar
    user_descriptions = []
    num_user_batches = (len(users) - 1) // BATCH_SIZE + 1
    
    print("\nProcessing Users...")
    for i in tqdm(range(0, len(users), BATCH_SIZE), total=num_user_batches):
        batch = users[i:i+BATCH_SIZE]
        desc, val_emb, det_emb = process_batch(batch, is_user=True)
        user_descriptions.extend(desc)
        user_values_embeddings.update(val_emb)
        user_details_embeddings.update(det_emb)

    # Process brands in batches with progress bar
    brand_descriptions = []
    num_brand_batches = (len(brands) - 1) // BATCH_SIZE + 1
    
    print("\nProcessing Brands...")
    for i in tqdm(range(0, len(brands), BATCH_SIZE), total=num_brand_batches):
        batch = brands[i:i+BATCH_SIZE]
        desc, val_emb, det_emb = process_batch(batch, is_user=False)
        brand_descriptions.extend(desc)
        brand_values_embeddings.update(val_emb)
        brand_details_embeddings.update(det_emb)

    print(f"\nGenerated embeddings for {len(user_values_embeddings)} users and {len(brand_values_embeddings)} brands")
    
    # Setup data directory and save embeddings
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)

    print("\nSaving embeddings to files...")
    np.save(os.path.join(data_dir, 'user_values_embeddings.npy'), user_values_embeddings)
    np.save(os.path.join(data_dir, 'user_details_embeddings.npy'), user_details_embeddings)
    np.save(os.path.join(data_dir, 'brand_values_embeddings.npy'), brand_values_embeddings)
    np.save(os.path.join(data_dir, 'brand_details_embeddings.npy'), brand_details_embeddings)

    return (user_descriptions, brand_descriptions, 
            user_values_embeddings, user_details_embeddings,
            brand_values_embeddings, brand_details_embeddings,
            [s.to_dict().get('minCompensation', 0) for s in users],
            [b.to_dict().get('maxBudget', 0) for b in brands])

def test_random_pair_similarities(num_samples=5):
    """
    Test similarity computation between random user-brand pairs.
    
    Args:
        num_samples (int): Number of random pairs to test
        
    Returns:
        list: Results containing similarity scores and profile information
    """
    data = test_generate_and_save_data()
    (user_descriptions, brand_descriptions,
     user_values_embeddings, user_details_embeddings,
     brand_values_embeddings, brand_details_embeddings,
     user_min_compensations, brand_max_budgets) = data
    
    user_ids = list(user_values_embeddings.keys())
    brand_ids = list(brand_values_embeddings.keys())
    
    results = []
    for _ in range(num_samples):
        user_id = random.choice(user_ids)
        brand_id = random.choice(brand_ids)
        
        user_idx = user_ids.index(user_id)
        brand_idx = brand_ids.index(brand_id)
        
        values_similarity, details_similarity, price_compatibility, overall_similarity = compute_match_similarity(
            user_values_embeddings[user_id],
            user_details_embeddings[user_id],
            brand_values_embeddings[brand_id],
            brand_details_embeddings[brand_id],
            user_min_compensations[user_idx],
            brand_max_budgets[brand_idx]
        )
        
        user_desc = next(u for u in user_descriptions if u['user_id'] == user_id)
        brand_desc = next(b for b in brand_descriptions if b['brand_id'] == brand_id)
        
        results.append({
            'values_similarity': values_similarity,
            'details_similarity': details_similarity,
            'overall_similarity': overall_similarity,
            'price_compatibility': price_compatibility,
            'user_values': user_desc['values_description'],
            'user_details': user_desc['details_description'],
            'brand_values': brand_desc['values_description'],
            'brand_details': brand_desc['details_description'],
            'user_min_compensation': user_min_compensations[user_idx],
            'brand_max_budget': brand_max_budgets[brand_idx]
        })
    
    return results

if __name__ == "__main__":
    # Add warning for production environment
    print("\nWARNING: Running tests in PRODUCTION environment!")
    response = input("Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Tests cancelled.")
        sys.exit(0)
        
    # Get results
    results = test_random_pair_similarities()
    
    # Create output string
    output = []
    output.append("\nTesting similarity with random samples...")
    for i, result in enumerate(results, 1):
        output.append(f"\nMatch {i}:")
        output.append(f"Overall Similarity Score: {result['overall_similarity']:.4f}")
        output.append(f"Values Similarity Score: {result['values_similarity']:.4f}")
        output.append(f"Details Similarity Score: {result['details_similarity']:.4f}")
        output.append(f"Price Compatibility: {result['price_compatibility']:.4f}")
        output.append(f"User Values: {result['user_values']}")
        output.append(f"User Details: {result['user_details']}")
        output.append(f"Brand Values: {result['brand_values']}")
        output.append(f"Brand Details: {result['brand_details']}")
        output.append(f"User Min Compensation: {result['user_min_compensation']}")
        output.append(f"Brand Max Budget: {result['brand_max_budget']}")
    
    # Print to console
    print('\n'.join(output))
    
    # Write to file
    output_file = os.path.join(os.path.dirname(__file__), 'random_pairs_results.txt')
    with open(output_file, 'w') as f:
        f.write('\n'.join(output))
    
    print(f"\nResults have been saved to: {output_file}")