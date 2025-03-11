import os
from google.cloud import firestore
import firebase_admin
from firebase_admin import credentials as firebase_credentials
import logging
from multiprocessing import Lock
from google.auth import default

logger = logging.getLogger('api.config.firebase')

class FirestoreClient:
    _instance = None
    _initialized = False
    _lock = Lock()

    def __new__(cls):
        try:
            if cls._instance is None:
                with cls._lock:
                    if cls._instance is None:
                        cls._instance = super(FirestoreClient, cls).__new__(cls)
            return cls._instance
        except Exception as e:
            print(f"Failed to create FirestoreClient instance: {str(e)}")
            raise

    def initialize(self):
        """Initialize Firestore client with appropriate credentials."""
        with self._lock:
            if self._initialized:
                return
                
            try:
                project_id = 'astn-mvp-v1'
                
                # Initialize Firebase Admin SDK
                if not firebase_admin._apps:
                    if os.getenv('K_SERVICE'):  # Cloud Run environment
                        app_credentials, project = default()
                        print(
                            f"Initializing Firebase Admin in Cloud Run with service account: "
                            f"{getattr(app_credentials, 'service_account_email', 'N/A')}"
                        )
                        
                        firebase_admin.initialize_app(None, {
                            'projectId': project_id,
                            'credential': firebase_admin.credentials.ApplicationDefault()
                        })
                    else:
                        cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
                        print(f"GOOGLE_APPLICATION_CREDENTIALS: {cred_path}")
                        if not cred_path:
                            print("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
                            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not set")
                        
                        if not os.path.exists(cred_path):
                            print(f"Credentials file not found at path: {cred_path}")
                            raise FileNotFoundError(f"Credentials file not found at path: {cred_path}")

                        cred = firebase_credentials.Certificate(cred_path)
                        firebase_admin.initialize_app(cred)
                        print("Initialized Firebase Admin with service account credentials")

                # Initialize Firestore client
                if os.getenv("USE_FIRESTORE_EMULATOR") == "true":
                    os.environ["FIRESTORE_EMULATOR_HOST"] = "127.0.0.1:8080"
                    self.db = firestore.Client(project=project_id)
                    print("Connected to Firestore Emulator")
                else:
                    if os.getenv('K_SERVICE'):
                        app_credentials, project = default()
                        self.db = firestore.Client(
                            project=project_id,
                            credentials=app_credentials
                        )
                        print("Connected to Firestore in Cloud Run environment")
                    else:
                        self.db = firestore.Client(project=project_id)
                        print("Connected to Firestore in development environment")

                self._initialized = True
                
            except Exception as e:
                print(f"Failed to initialize Firestore: {str(e)}")
                raise

def initialize_firebase():
    """Initialize Firebase for the current process."""
    try:
        client = FirestoreClient()
        client.initialize()
        return client.db
    except Exception as e:
        print(f"Failed to initialize Firebase in process {os.getpid()}: {str(e)}")
        raise

# Initialize the db instance when the module loads
db = initialize_firebase()
