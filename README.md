# Recommendation API with Sentence Transformers

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-3.3.1-green.svg)](https://www.sbert.net/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-red.svg)](https://flask.palletsprojects.com/)
[![Firebase](https://img.shields.io/badge/Firebase-Admin-orange.svg)](https://firebase.google.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

A production-ready recommendation system powered by state-of-the-art Sentence Transformers (BERT) for intelligent user-brand matching. The system leverages advanced Natural Language Processing to create semantic embeddings that enable precise matching based on:

- Semantic similarity through BERT-based embedding analysis
- Multi-dimensional profile matching using transformer encodings
- Dynamic threshold adjustment for compatibility scoring
- Real-time embedding updates through custom ML pipeline

## Core Features

### Transformer-Based Embedding Generation
- BERT-based Sentence Transformers for semantic text encoding
- Custom ML pipeline for profile vector generation
- Dynamic embedding updates based on interactions
- Real-time semantic analysis of profile data
- Automated vector computation with error handling

### Advanced Matching System
- Multi-dimensional semantic similarity computation
- Sophisticated compatibility analysis with dynamic thresholds
- Real-time embedding updates based on interactions
- Customizable matching parameters and weights
- Intelligent price compatibility algorithms

### Implementation Examples

```python
# Generate embeddings using Sentence Transformers
compute_initial_user_embedding(user_id, db, model)
compute_initial_brand_embedding(brand_id, db, model)

# Compute semantic similarity
def compute_match_similarity(
    user_values_embedding, 
    user_details_embedding,
    brand_values_embedding, 
    brand_details_embedding,
    user_min_compensation,
    brand_max_budget
):
    """
    Computes similarity using:
    - BERT-based semantic similarity
    - Multi-dimensional embedding comparison
    - Dynamic price compatibility
    """

# Update embeddings with interaction data
update_user_embeddings(user_id, brand_id, db)
```

## System Architecture

### Project Structure
```
recommendation_model/
├── api/                      # API implementation
│   ├── config/              # Configuration management
│   │   └── firebase_config.py
│   ├── models/              # Transformer models and algorithms
│   │   └── finalmodel.py    # Core matching logic
│   └── app.py              # Flask API endpoints
├── tests/                   # Test suite
│   ├── data/               # Test data and embeddings
│   ├── results/            # Test results and analysis
│   ├── scripts/            # Data generation utilities
│   └── unit/               # Unit test modules
└── docs/                   # Documentation
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/compute-user-embedding` | POST | Generate user embeddings using BERT |
| `/get-brand-matches` | GET | Retrieve semantically matched brands |
| `/update-user-embedding` | POST | Update user embeddings |
| `/compute-brand-embedding` | POST | Generate brand embeddings |

## Setup and Development

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Firebase Configuration
1. Create a Firebase project and download service account credentials
2. Configure environment variables:
```bash
# .env
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/firebase/credentials.json
```

### Development Environment
```bash
# Install Firebase CLI
npm install -g firebase-tools

# Initialize Firebase project
firebase init

# Start local emulator
firebase emulators:start
```

## Testing

The project includes comprehensive testing:

```bash
# Run all tests
python -m pytest tests/

# Run specific test suite
python -m pytest tests/unit/
```

Key test components:
- Unit tests for transformer operations
- Integration tests for API endpoints
- Embedding accuracy validation
- Performance benchmarks

## Dependencies

### Core Dependencies
- sentence-transformers>=3.3.1
- numpy>=2.1.3
- pandas>=2.2.3
- scikit-learn>=1.5.2
- firebase-admin>=6.6.0
- flask>=3.0.0
- python-dotenv>=1.0.1

### Development Dependencies
- pytest>=8.0.0
- matplotlib>=3.9.2
- tqdm>=4.66.1

## Security and Best Practices

### Security Implementation
- Environment-based credential management
- Request authentication and validation
- Secure data storage practices
- Rate limiting and error handling

### Production Guidelines
- Never commit sensitive files to version control
- Use environment variables for configuration
- Follow Firebase security best practices
- Implement proper error handling and logging

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the test suite
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.