#!/usr/bin/env python3
"""
Main script to run the CSAO Rail Recommendation System
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Custom modules
from features import create_features
from models import train_and_evaluate
from cold_start import handle_cold_start
from business_impact import simulate_business_impact
from ab_testing import simulate_ab_test

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

def main():
    print("=== CSAO Rail Recommendation System ===\n")

    # Load data
    print("Loading data...")
    orders_df = pd.read_csv('data/orders.csv')
    users_df = pd.read_csv('data/users.csv')
    restaurants_df = pd.read_csv('data/restaurants.csv')
    items_df = pd.read_csv('data/items.csv')

    print(f"Orders shape: {orders_df.shape}")
    print(f"Users shape: {users_df.shape}")
    print(f"Restaurants shape: {restaurants_df.shape}")
    print(f"Items shape: {items_df.shape}\n")

    # Data exploration
    print("Target distribution:")
    print(orders_df['addon_accepted'].value_counts(normalize=True))
    print()

    # Feature engineering
    print("Creating features...")
    orders_df = create_features(orders_df)
    print("Features created.\n")

    # Model training and evaluation
    print("Training and evaluating models...")
    results, lr_model, gb_model = train_and_evaluate(orders_df)

    print("Model Performance:")
    for model_name, metrics in results.items():
        print(f"{model_name}: AUC={metrics['auc']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, Precision@5={metrics['precision_at_5']:.3f}")
    print()

    # Cold start handling
    print("Testing cold start handling...")
    user_history = orders_df['user_id'].unique()
    cold_start_prob = handle_cold_start(9999, 201, 'New York', 'Italian', 12, 'Premium', 'Main', orders_df, user_history)
    print(f"Cold start probability for new user: {cold_start_prob:.3f}\n")

    # Business impact simulation
    print("Simulating business impact...")
    baseline_auc = results['baseline']['auc']
    model_auc = results['gradient_boosting']['auc']
    impact = simulate_business_impact(baseline_auc, model_auc)

    print("Business Impact:")
    for key, value in impact.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    print()

    # A/B testing simulation
    print("Simulating A/B testing...")
    ab_results = simulate_ab_test(baseline_auc, model_auc)

    print("A/B Test Results:")
    for key, value in ab_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    print()

    print("=== Project completed successfully! ===")

if __name__ == "__main__":
    main()
