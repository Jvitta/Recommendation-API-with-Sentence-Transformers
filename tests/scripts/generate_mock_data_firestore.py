import sys
import os
import random
from dotenv import load_dotenv

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)  # Add the project root instead of api path

# Load environment variables
load_dotenv()

# Check for credentials
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    raise EnvironmentError("Please set GOOGLE_APPLICATION_CREDENTIALS environment variable")

# Import Firestore config after adding to path
from api.config.firebase_config import initialize_firebase, db

# Load environment variables
load_dotenv()

# Initialize Firestore (will use development project_id from firebase_config.py)
if db is None:
    db = initialize_firebase()

# Random data lists (copied from generate_mock_data.py)
user_types = ["Pro", "Semi-pro", "College", "High School", "Retired", "Amateur", "Paralympic", "Olympic", "eSports"]
sports = ["Basketball", "Soccer", "Tennis", "Swimming", "Track and Field", "Golf", "Baseball", "Hockey", "Boxing"]
values_interests = [
    "Family and Relationships", "Education", "Music", "Technology", "Innovation", 
    "Health and Wellness", "Travel", "Philanthropy", "Art and Culture", "Fitness", 
    "Entrepreneurship", "Sustainability", "Animal Welfare", "Community Engagement", 
    "Adventure and Outdoor Activities", "Food and Cooking", "Fashion", "Mental Health", 
    "Gaming", "Environmental Conservation"
]
locations = ["New York", "Los Angeles", "Chicago", "Houston", "San Francisco", "Boston", "Seattle", "Miami"]
teams = ["Team A", "Team B", "Team C", "Team D", "Team E"]

industries = ["Sportswear", "Health Products", "Fitness Equipment", "Technology", "Travel", "Education"]
company_sizes = ["Small", "Medium", "Large"]
countries = ["USA", "Canada", "UK", "Germany", "Australia"]
states = ["California", "Texas", "New York", "Florida", "Illinois"]

def generate_random_user():
    return {
        "userType": random.choice(user_types),
        "category": random.choice(sports),
        "valuesInterests": random.sample(values_interests, k=random.randint(0, 10)),
        "location": random.choice(locations),
        "affiliation": random.choice(teams) if random.random() > 0.5 else None,
        "minCompensation": random.randint(500, 5000)
    }

def generate_random_brand():
    return {
        "name": f"Brand {random.randint(1, 1000)}",
        "industry": random.choice(industries),
        "companyValues": random.sample(values_interests, k=random.randint(0, 10)),
        "companySize": random.choice(company_sizes),
        "country": random.choice(countries),
        "state": random.choice(states) if random.random() > 0.5 else None,
        "maxBudget": random.randint(2000, 10000)
    }

def add_mock_data_to_firestore(num_users=100, num_brands=100):
    try:
        # Add users
        for i in range(num_users):
            user_data = generate_random_user()
            user_ref = db.collection('Users').document()
            user_ref.set(user_data)
            print(f"Added user {i+1}/{num_users} with ID: {user_ref.id}")

        # Add brands
        for i in range(num_brands):
            brand_data = generate_random_brand()
            brand_ref = db.collection('Brands').document()  # Auto-generate ID
            brand_ref.set(brand_data)
            print(f"Added brand {i+1}/{num_brands} with ID: {brand_ref.id}")

        return True
    except Exception as e:
        print(f"Error adding mock data to Firestore: {str(e)}")
        return False

def verify_data():
    try:
        # Check Users collection
        users = db.collection('Users').get()
        users_list = list(users)
        print(f"\nFound {len(users_list)} users in Users collection")

        # Check Brands collection
        brands = db.collection('Brands').get()
        brands_list = list(brands)
        print(f"Found {len(brands_list)} brands in Brands collection")

        if len(users_list) == 0 and len(brands_list) == 0:
            print("WARNING: No data found in either collection!")
            return

        # Print first document from each collection as sample
        if users_list:
            print("\nSample user data:")
            print(f"Document ID: {users_list[0].id}")
            print(f"Data: {users_list[0].to_dict()}")

        if brands_list:
            print("\nSample brand data:")
            print(f"Document ID: {brands_list[0].id}")
            print(f"Data: {brands_list[0].to_dict()}")

    except Exception as e:
        print(f"Error verifying data: {str(e)}")

if __name__ == "__main__":
    print(f"Firestore Emulator Host: {os.getenv('FIRESTORE_EMULATOR_HOST')}")
    print("Adding mock data to Firestore emulator...")
    success = add_mock_data_to_firestore()
    
    if success:
        print("\nMock data added successfully!")
        print("\nVerifying data...")
        verify_data()
    else:
        print("\nFailed to add mock data!")
