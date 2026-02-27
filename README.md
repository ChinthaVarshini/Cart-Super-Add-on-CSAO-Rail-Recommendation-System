Cart Super Add-On (CSAO) Rail Recommendation System

An end-to-end machine learning system that recommends contextual add-on items (e.g., desserts, beverages, sides) during a food ordering session to improve Average Order Value (AOV) and user experience.

This project was developed as a hackathon solution demonstrating data simulation, modeling, evaluation, and production-level system design.

🎯 Objective

Build a scalable, real-time recommendation system that:

Suggests relevant add-on items based on cart composition

Adapts to user behavior and contextual signals (time, city, segment)

Handles cold-start users and restaurants

Operates within strict latency constraints (<300ms)

Demonstrates measurable business impact

🧠 Problem Formulation

The task is framed as:

Binary Classification
Predict whether a user will accept an add-on recommendation.

Ranking Problem
Rank candidate add-ons by likelihood of acceptance to generate Top-N recommendations.

📊 Dataset

Since no dataset was provided, a synthetic dataset was generated to simulate realistic food delivery behavior.

Dataset Summary

1000 users across 10 cities

200 restaurants across multiple cuisines

2000 food items categorized as Main, Side, Dessert, Drink

10,000 simulated orders

Features Included

User Features

City

Segment (Premium/Budget)

Order frequency

Average order value

Restaurant Features

Cuisine type

Rating

Cart Context

Item categories

Cart size

Mealtime flags (breakfast/lunch/dinner)

Hour of day (sin/cos encoded)

Target Variable

addon_accepted (binary)

Realistic Behavioral Simulation

Higher acceptance during lunch/dinner hours

Premium users more likely to accept add-ons

Desserts/drinks more likely after main dishes

Sparse history users included

Cold-start users and restaurants simulated

⚙️ System Architecture
Feature Engineering

Temporal encoding (sin/cos hour encoding)

User segmentation features

Restaurant normalization

Cart category context

Categorical encodings

Models Implemented

Baseline Model – Average acceptance probability

Logistic Regression – Interpretable linear model

Gradient Boosting (XGBoost) – Captures nonlinear interactions

Evaluation Metrics

ROC-AUC

Precision

Recall

Precision@K

Ranking-based evaluation

❄ Cold Start Strategy

New User → City-level popularity fallback

New Restaurant → Cuisine-level fallback

Sparse History → Context-heavy feature reliance

Rule-based backup heuristics

🚀 Production Design
Cart Event → Feature Store → Model API → Ranking Engine → Response API
Key Properties

Inference latency < 300ms

Scalable deployment using containerized services

Redis-based feature caching

Weekly retraining pipeline

A/B testing integration

📈 Business Impact Simulation

Projected improvements based on model lift:

Increase in Add-on Acceptance Rate

AOV lift

Improvement in Cart-to-Order ratio

Higher average items per order

🧪 A/B Testing Framework

Control: Baseline recommendation logic

Treatment: ML-based ranking model

Guardrail metrics to prevent UX degradation

Statistical testing for significance

📂 Project Structure
.
├── data/
├── notebooks/
├── src/
├── requirements.txt
└── README.md
🔮 Future Enhancements

Real-time feature store integration

Contextual bandit-based exploration

Deeper personalization

Monitoring & drift detection

Production-grade API deployment

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
