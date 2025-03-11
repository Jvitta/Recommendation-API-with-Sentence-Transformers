import os
import sys
from sentence_transformers import SentenceTransformer

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Set credentials for production database
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/jvitt/OneDrive/Documents/.google/credentials/credentials_prod.json"

# Import Firestore config after adding to path
from api.config.firebase_config import initialize_firebase, db
from api.models.finalmodel import compute_initial_brand_embedding

# Initialize Firestore (will use development project_id from firebase_config.py)
if db is None:
    db = initialize_firebase()

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_brands():
    brands = [
        {"name": "Lululemon", "industry": "Athletic Apparel", "companyValues": ["Fitness", "Sustainability", "Innovation"], "companySize": "Large", "country": "Canada", "state": "", "maxBudget": 500000},
        {"name": "Old Navy", "industry": "Retail Clothing", "companyValues": ["Fashion", "Community Engagement", "Philanthropy"], "companySize": "Large", "country": "USA", "state": "California", "maxBudget": 300000},
        {"name": "Poppi", "industry": "Beverage", "companyValues": ["Health and Wellness", "Innovation"], "companySize": "Small", "country": "USA", "state": "Texas", "maxBudget": 50000},
        {"name": "Native", "industry": "Personal Care", "companyValues": ["Sustainability", "Health and Wellness", "Environmental Conservation"], "companySize": "Medium", "country": "USA", "state": "California", "maxBudget": 100000},
        {"name": "Harrys", "industry": "Men's Grooming", "companyValues": ["Innovation", "Sustainability", "Philanthropy"], "companySize": "Medium", "country": "USA", "state": "New York", "maxBudget": 150000},
        {"name": "Uniqlo", "industry": "Fast Fashion", "companyValues": ["Fashion", "Innovation", "Sustainability"], "companySize": "Large", "country": "Japan", "state": "", "maxBudget": 400000},
        {"name": "PacificCycle", "industry": "Bicycle Manufacturing", "companyValues": ["Adventure and Outdoor Activities", "Sustainability", "Community Engagement"], "companySize": "Medium", "country": "USA", "state": "Wisconsin", "maxBudget": 75000},
        {"name": "Pacsun", "industry": "Youth Fashion Retail", "companyValues": ["Fashion", "Community Engagement", "Art and Culture"], "companySize": "Medium", "country": "USA", "state": "California", "maxBudget": 120000},
        {"name": "Fabletics", "industry": "Athleisure", "companyValues": ["Fitness", "Fashion", "Community Engagement"], "companySize": "Large", "country": "USA", "state": "California", "maxBudget": 250000},
        {"name": "Gymshark", "industry": "Fitness Apparel", "companyValues": ["Fitness", "Entrepreneurship", "Community Engagement"], "companySize": "Medium", "country": "UK", "state": "", "maxBudget": 200000},
        {"name": "Gatorade", "industry": "Sports Drinks", "companyValues": ["Fitness", "Innovation", "Health and Wellness"], "companySize": "Large", "country": "USA", "state": "Illinois", "maxBudget": 600000},
        {"name": "Oura", "industry": "Wearable Technology", "companyValues": ["Technology", "Health and Wellness", "Innovation"], "companySize": "Medium", "country": "Finland", "state": "", "maxBudget": 100000},
        {"name": "Chime", "industry": "Financial Technology", "companyValues": ["Technology", "Innovation", "Community Engagement"], "companySize": "Medium", "country": "USA", "state": "California", "maxBudget": 150000},
        {"name": "Fila", "industry": "Sportswear", "companyValues": ["Fashion", "Fitness"], "companySize": "Large", "country": "South Korea", "state": "", "maxBudget": 350000},
        {"name": "Recess", "industry": "Functional Beverages", "companyValues": ["Health and Wellness", "Innovation", "Mental Health"], "companySize": "Small", "country": "USA", "state": "New York", "maxBudget": 50000},
        {"name": "Champion", "industry": "Athletic Apparel", "companyValues": ["Fitness", "Fashion"], "companySize": "Large", "country": "USA", "state": "North Carolina", "maxBudget": 400000},
        {"name": "Reebok", "industry": "Sportswear", "companyValues": ["Fitness", "Innovation", "Fashion"], "companySize": "Large", "country": "USA", "state": "Massachusetts", "maxBudget": 500000},
        {"name": "Redbull", "industry": "Energy Drinks", "companyValues": ["Adventure and Outdoor Activities", "Innovation", "Community Engagement"], "companySize": "Large", "country": "Austria", "state": "", "maxBudget": 700000},
        {"name": "LiquidIV", "industry": "Hydration Products", "companyValues": ["Health and Wellness", "Sustainability", "Philanthropy"], "companySize": "Medium", "country": "USA", "state": "California", "maxBudget": 100000},
        {"name": "CapitalOne", "industry": "Banking", "companyValues": ["Technology", "Innovation", "Community Engagement"], "companySize": "Large", "country": "USA", "state": "Virginia", "maxBudget": 300000}
    ]
    return brands


def add_brands_to_firestore(brands):
    try:
        for i, brand_data in enumerate(brands):
            # Create Firestore document reference
            brand_ref = db.collection('Brands').document()
            
            # Store initial brand data in Firestore to create the document
            brand_ref.set(brand_data)
            
            # Compute initial embeddings for the brand
            embeddings_result = compute_initial_brand_embedding(brand_ref.id, db, model)
            
            # Add embeddings to brand data
            brand_data['embeddings'] = {
                'values_embedding': embeddings_result['values_embedding'],
                'details_embedding': embeddings_result['details_embedding']
            }

            # Update brand data in Firestore with embeddings
            brand_ref.update(brand_data)
            print(f"Added brand {i+1}/{len(brands)} with ID: {brand_ref.id}")
    except Exception as e:
        print(f"Error adding brands to Firestore: {str(e)}")

if __name__ == "__main__":
    brands = generate_brands()
    add_brands_to_firestore(brands)
