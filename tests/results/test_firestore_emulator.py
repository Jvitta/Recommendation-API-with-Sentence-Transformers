import sys
import os
from dotenv import load_dotenv

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import Firestore config after adding to path
from api.config.firebase_config import initialize_firestore, db

# Load environment variables and initialize Firestore emulator
load_dotenv()
os.environ["USE_FIRESTORE_EMULATOR"] = "true"
if db is None:
    db = initialize_firestore(project_root)

def test_firestore_connection():
    try:
        # Try to create a test collection and document
        test_ref = db.collection('test').document('test_doc')
        test_ref.set({
            'message': 'Hello from emulator!',
            'timestamp': 123456789
        })

        # Try to read the document back
        doc = test_ref.get()
        if doc.exists:
            print("Successfully wrote and read from Firestore emulator!")
            print(f"Document data: {doc.to_dict()}")
        else:
            print("Document was not created successfully.")

        # Clean up - delete the test document
        test_ref.delete()
        print("Test document cleaned up successfully.")
        
        return True
    except Exception as e:
        print(f"Error testing Firestore emulator: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_firestore_connection()
    if success:
        print("\nFirestore emulator test completed successfully!")
    else:
        print("\nFirestore emulator test failed!")
