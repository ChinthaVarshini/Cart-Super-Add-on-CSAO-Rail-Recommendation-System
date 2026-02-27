import pandas as pd
import numpy as np

def new_user_fallback(city, orders_df):
    """
    For new users, use city-level popularity.
    """
    city_orders = orders_df[orders_df['city'] == city]
    if city_orders.empty:
        return 0.2  # Global average
    return city_orders['addon_accepted'].mean()

def new_restaurant_fallback(cuisine, orders_df):
    """
    For new restaurants, use cuisine-level average.
    """
    cuisine_orders = orders_df[orders_df['cuisine'] == cuisine]
    if cuisine_orders.empty:
        return 0.2
    return cuisine_orders['addon_accepted'].mean()

def rule_based_fallback(hour, segment, category):
    """
    Rule-based heuristic.
    """
    prob = 0.2
    if 11 <= hour <= 14 or 18 <= hour <= 21:
        prob += 0.1
    if segment == 'Premium':
        prob += 0.1
    if category in ['Dessert', 'Drink']:
        prob += 0.05
    return min(prob, 1.0)

def handle_cold_start(user_id, restaurant_id, city, cuisine, hour, segment, category, orders_df, user_history):
    """
    Handle cold start scenarios.
    """
    if user_id not in user_history:
        # New user
        return new_user_fallback(city, orders_df)
    elif restaurant_id not in orders_df['restaurant_id'].unique():
        # New restaurant
        return new_restaurant_fallback(cuisine, orders_df)
    else:
        # Rule-based for other cases
        return rule_based_fallback(hour, segment, category)
