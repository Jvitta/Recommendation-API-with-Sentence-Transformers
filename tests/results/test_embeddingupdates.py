import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
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
from api.models.finalmodel import compute_match_similarity, update_user_embeddings
from tests.results.test_embeddings import generate_and_save_data

# Load environment variables and initialize Firestore
db = initialize_firebase()

def test_embedding_update_process(num_clusters=5, alpha=0.7, test_number=1):
    """
    Test the embedding update process using clustering and similarity analysis.
    
    Args:
        num_clusters (int): Number of clusters for k-means
        alpha (float): Weight for exponential moving average
        test_number (int): Test iteration number
        
    Returns:
        tuple: Results dictionary and output messages
    """
    print(f"\nStarting test {test_number} with {num_clusters} clusters...")
    
    # Get all the data
    (user_descriptions, brand_descriptions,
     user_values_embeddings, user_details_embeddings,
     brand_values_embeddings, brand_details_embeddings,
     user_min_compensations, brand_max_budgets) = generate_and_save_data()
    
    # Combine embeddings for clustering
    brand_embeddings = []
    for brand_id in brand_values_embeddings:
        combined = np.concatenate((
            brand_values_embeddings[brand_id],
            brand_details_embeddings[brand_id]
        ))
        brand_embeddings.append(combined)
    brand_embeddings = np.array(brand_embeddings)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    brand_clusters = kmeans.fit_predict(brand_embeddings)
    
    # Choose a random user
    user_id = random.choice(list(user_values_embeddings.keys()))
    user_desc = next(u for u in user_descriptions if u['user_id'] == user_id)
    user_embedding = np.concatenate((user_values_embeddings[user_id],
        user_details_embeddings[user_id]))
    
    print(f"Selected user ID: {user_id}")
    
    # Find closest cluster
    distances_to_clusters = cdist([user_embedding], kmeans.cluster_centers_, 'euclidean')[0]
    target_cluster = np.argmin(distances_to_clusters)
    
    # Get brands in target cluster
    target_cluster_brand_ids = []
    for i, cluster in enumerate(brand_clusters):
        if cluster == target_cluster:
            brand_id = list(brand_values_embeddings.keys())[i]
            target_cluster_brand_ids.append(brand_id)
    
    num_brands_in_cluster = len(target_cluster_brand_ids)
    print(f"Found {num_brands_in_cluster} brands in closest cluster")
    
    if num_brands_in_cluster < 2:
        print("Not enough brands in cluster for meaningful test")
        return None

    # Select brands for interaction and testing
    num_interactions = min(5, num_brands_in_cluster // 2)
    num_test_brands = num_interactions
    total_brands_needed = num_interactions + num_test_brands

    selected_brand_ids = random.sample(target_cluster_brand_ids, total_brands_needed)
    interacted_brand_ids = selected_brand_ids[:num_interactions]
    test_brand_ids = selected_brand_ids[num_interactions:]

    # Calculate initial similarities
    print("Calculating initial similarities...")
    initial_similarities = []
    for brand_id in test_brand_ids:
        brand_idx = [b['brand_id'] for b in brand_descriptions].index(brand_id)
        values_sim, details_sim, price_comp, overall_sim = compute_match_similarity(
            user_values_embeddings[user_id],
            user_details_embeddings[user_id],
            brand_values_embeddings[brand_id],
            brand_details_embeddings[brand_id],
            user_min_compensations[[u['user_id'] for u in user_descriptions].index(user_id)],
            brand_max_budgets[brand_idx]
        )
        initial_similarities.append(overall_sim)

    initial_avg_similarity = np.mean(initial_similarities)

    # Update embeddings through interactions using transactions
    print("Updating embeddings through interactions...")
    for brand_id in tqdm(interacted_brand_ids, desc="Processing interactions"):
        try:
            # Use the new transaction-based update function
            result = update_user_embeddings(user_id, brand_id, db, alpha)
            if not result.get('success', False):
                print(f"Warning: Failed to update embeddings for interaction with brand {brand_id}")
        except Exception as e:
            print(f"Error updating embeddings for brand {brand_id}: {str(e)}")
            continue

    # Get updated embeddings from Firestore
    updated_user = db.collection('Users').document(user_id).get().to_dict()
    current_values_embedding = np.array(updated_user['embeddings']['values_embedding'])
    current_details_embedding = np.array(updated_user['embeddings']['details_embedding'])

    # Calculate final similarities
    print("Calculating final similarities...")
    final_similarities = []
    for brand_id in test_brand_ids:
        brand_idx = [b['brand_id'] for b in brand_descriptions].index(brand_id)
        values_sim, details_sim, price_comp, overall_sim = compute_match_similarity(
            current_values_embedding,
            current_details_embedding,
            brand_values_embeddings[brand_id],
            brand_details_embeddings[brand_id],
            user_min_compensations[[u['user_id'] for u in user_descriptions].index(user_id)],
            brand_max_budgets[brand_idx]
        )
        final_similarities.append(overall_sim)

    final_avg_similarity = np.mean(final_similarities)

    # Visualize results with PCA
    pca = PCA(n_components=2)
    brand_embeddings_2d = pca.fit_transform(brand_embeddings)
    user_embedding_2d = pca.transform([user_embedding])[0]
    updated_user_embedding = np.concatenate((current_values_embedding, current_details_embedding))
    updated_user_embedding_2d = pca.transform([updated_user_embedding])[0]

    # Create plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, num_clusters))
    
    for cluster in range(num_clusters):
        mask = brand_clusters == cluster
        plt.scatter(brand_embeddings_2d[mask, 0], brand_embeddings_2d[mask, 1],
                   color=colors[cluster], alpha=0.6, label=f'Cluster {cluster}')
    
    plt.scatter(user_embedding_2d[0], user_embedding_2d[1],
               color='black', marker='X', s=200, label='Initial User Embedding')
    plt.scatter(updated_user_embedding_2d[0], updated_user_embedding_2d[1],
               color='red', marker='X', s=200, label='Updated User Embedding')

    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Brand Clusters and User Embeddings')
    plt.legend()

    # Save results
    results = {
        'test_number': test_number,
        'user_values': user_desc['values_description'],
        'user_details': user_desc['details_description'],
        'num_interactions': num_interactions,
        'initial_avg_similarity': initial_avg_similarity,
        'final_avg_similarity': final_avg_similarity,
        'similarity_change': final_avg_similarity - initial_avg_similarity
    }

    # Create output string
    output = []
    output.append(f"\nTest {test_number} Results:")
    output.append(f"Number of interactions: {num_interactions}")
    output.append(f"Initial average similarity: {initial_avg_similarity:.4f}")
    output.append(f"Final average similarity: {final_avg_similarity:.4f}")
    output.append(f"Change in similarity: {results['similarity_change']:.4f}")
    output.append("\nUser Profile:")
    output.append(f"Values: {results['user_values']}")
    output.append(f"Details: {results['user_details']}")

    # Save plot
    plot_file = os.path.join(os.path.dirname(__file__), f'embedding_update_test_{test_number}.png')
    plt.savefig(plot_file)
    plt.close()

    return results, output

if __name__ == "__main__":
    # Run multiple tests
    all_results = []
    for i in range(1, 6):
        print(f"\nRunning test {i}/5...")
        results, output = test_embedding_update_process(test_number=i)
        all_results.append((results, output))

    # Create output string for all tests
    output = []
    for results, output_str in all_results:
        output.append(f"\n\nTest {results['test_number']} Results:")
        output.append(f"Number of interactions: {results['num_interactions']}")
        output.append(f"Initial average similarity: {results['initial_avg_similarity']:.4f}")
        output.append(f"Final average similarity: {results['final_avg_similarity']:.4f}")
        output.append(f"Change in similarity: {results['similarity_change']:.4f}")
        output.append("\nUser Profile:")
        output.append(f"Values: {results['user_values']}")
        output.append(f"Details: {results['user_details']}")
        output.append(f"\nPCA plot saved to: {results['plot_file']}")

        # Append additional output string
        output.append(output_str)

    # Write results to file
    output_file = os.path.join(os.path.dirname(__file__), 'embedding_updates_results_all_tests.txt')
    with open(output_file, 'w') as f:
        f.write('\n'.join(output))

    print(f"\nResults for all tests have been saved to: {output_file}")
