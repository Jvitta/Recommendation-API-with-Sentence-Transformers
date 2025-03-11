import sys
import os
import unittest
from datetime import datetime

# Add the api directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
api_path = os.path.join(project_root, 'api')
sys.path.append(api_path)

# Load and check environment variables
from dotenv import load_dotenv
load_dotenv()

if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    raise EnvironmentError("Please set GOOGLE_APPLICATION_CREDENTIALS environment variable")

from api.config.firebase_config import initialize_firestore

class TestFirestoreConnection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize Firestore client before running tests"""
        cls.db = initialize_firestore()
        cls.test_user_id = f"test_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cls.test_brand_id = f"test_brand_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def test_add_and_verify_data(self):
        """Test adding one user and one brand to Firestore"""
        try:
            # Create test user data
            user_data = {
                "userType": "Professional",
                "category": "Sports",
                "valuesInterests": ["Community", "Education"],
                "location": "New York",
                "affiliation": "Local Team",
                "minCompensation": 1000
            }

            # Create test brand data
            brand_data = {
                "name": "Test Brand",
                "industry": "Retail",
                "companyValues": ["Innovation", "Sustainability"],
                "companySize": "Medium",
                "country": "USA",
                "state": "New York",
                "maxBudget": 5000
            }

            # Add user to Firestore
            user_ref = self.db.collection('Users').document(self.test_user_id)
            user_ref.set(user_data)
            print(f"\nAdded test user with ID: {self.test_user_id}")

            # Add brand to Firestore
            brand_ref = self.db.collection('Brands').document(self.test_brand_id)
            brand_ref.set(brand_data)
            print(f"Added test brand with ID: {self.test_brand_id}")

            # Verify user data
            stored_user = user_ref.get()
            self.assertTrue(stored_user.exists)
            stored_user_data = stored_user.to_dict()
            self.assertEqual(stored_user_data["userType"], "Professional")
            self.assertEqual(stored_user_data["location"], "New York")

            # Verify brand data
            stored_brand = brand_ref.get()
            self.assertTrue(stored_brand.exists)
            stored_brand_data = stored_brand.to_dict()
            self.assertEqual(stored_brand_data["name"], "Test Brand")
            self.assertEqual(stored_brand_data["industry"], "Retail")

            print("\nVerification successful!")
            print(f"User data: {stored_user_data}")
            print(f"Brand data: {stored_brand_data}")

        except Exception as e:
            self.fail(f"Test failed with error: {str(e)}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test data after running tests"""
        try:
            # Delete test documents
            cls.db.collection('Users').document(cls.test_user_id).delete()
            cls.db.collection('Brands').document(cls.test_brand_id).delete()
            print(f"\nCleaned up test data (IDs: {cls.test_user_id}, {cls.test_brand_id})")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

if __name__ == '__main__':
    unittest.main()
