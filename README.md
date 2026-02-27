# Cart Super Add-On (CSAO) Rail Recommendation System

A machine learning system for recommending add-ons in a food delivery platform, designed for hackathon submission.

## Project Overview

This project implements an end-to-end ML pipeline for predicting and recommending add-ons (e.g., desserts, drinks) to users during their food ordering process. The system focuses on food delivery domain, incorporating city-wise behaviors, mealtime patterns, and cold start scenarios.

## Problem Formulation

- **Binary Classification**: Predict whether a user will accept an add-on recommendation.
- **Ranking Problem**: Rank potential add-ons by likelihood of acceptance for personalized recommendations.

## Dataset

Synthetic dataset simulating realistic food delivery behavior:

- **1000 users** across 10 cities
- **200 restaurants** with various cuisines
- **2000 food items** categorized as Main, Side, Dessert, Drink
- **10000 orders** with comprehensive features

### Key Features
- User demographics: city, segment (Premium/Budget), order frequency, average order value
- Restaurant info: cuisine, rating
- Order context: hour of day, mealtime flags, cart size, item category
- Target: addon_accepted (binary)

### Realistic Simulations
- Higher add-on acceptance during lunch/dinner hours
- Premium users more likely to accept add-ons
- Dessert/Drink add-ons more probable after Main items
- Sparse history for some users
- Cold start scenarios for new users/restaurants

## Architecture

### Feature Engineering
- Time-based features (sin/cos encoding of hour)
- User behavior features (premium flag)
- Restaurant rating normalization
- Cart context features (item categories)
- Categorical encodings (city, cuisine, category)

### Models
- **Baseline**: Predict average acceptance probability
- **Logistic Regression**: Linear model for interpretability
- **Gradient Boosting**: XGBoost for complex patterns

### Evaluation Metrics
- ROC-AUC
- Precision, Recall
- Precision@5 (top 5% predictions)

### Cold Start Strategies
- New user: City-level popularity fallback
- New restaurant: Cuisine-level average fallback
- Rule-based heuristics for edge cases

### Production Architecture
```
Cart Event → Feature Store → Model API → Ranking Engine → API Response
```

- **Inference Time**: < 300ms using optimized models and caching
- **Scalability**: Kubernetes-based deployment for millions of requests
- **Caching**: Redis for user features and popular recommendations
- **Retraining**: Weekly batch updates with A/B testing

## Business Impact

Simulated impact based on model performance:
- Add-on acceptance lift
- Revenue increase from additional add-ons
- Average Order Value (AOV) improvement
- Cart-to-order ratio enhancement

## A/B Testing Framework

- Control group: Baseline recommendations
- Treatment group: ML model recommendations
- Guardrail metrics: Ensure no negative impact on core metrics
- Statistical significance testing using t-tests

## Installation & Setup

1. Clone the repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the main notebook:
   ```bash
   jupyter notebook notebooks/CSAO_Rail_Recommendation.ipynb
   ```

## Project Structure

```
.
├── data/
│   ├── generate_data.py    # Synthetic data generation
│   ├── orders.csv          # Generated orders data
│   ├── users.csv           # User data
│   ├── restaurants.csv     # Restaurant data
│   └── items.csv           # Item data
├── notebooks/
│   └── CSAO_Rail_Recommendation.ipynb  # Main runnable notebook
├── src/
│   ├── features.py         # Feature engineering
│   ├── models.py           # Model training & evaluation
│   ├── cold_start.py       # Cold start handling
│   ├── business_impact.py  # Impact simulation
│   └── ab_testing.py       # A/B testing framework
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Future Improvements

- Implement real-time feature store (e.g., Feast)
- Add more sophisticated ranking algorithms
- Incorporate user feedback loops
- Expand to multi-armed bandit for exploration
- Deploy as microservice with FastAPI
- Add monitoring and alerting for production

## License

This project is for educational/hackathon purposes.
