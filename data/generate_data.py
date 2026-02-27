import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Parameters
NUM_USERS = 1000
NUM_RESTAURANTS = 200
NUM_ITEMS = 2000
NUM_ORDERS = 10000

# Cities
CITIES = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']

# Cuisines
CUISINES = ['Italian', 'Chinese', 'Mexican', 'Indian', 'American', 'Japanese', 'Thai', 'French', 'Greek', 'Korean']

# Categories
CATEGORIES = ['Main', 'Side', 'Dessert', 'Drink']

def generate_users():
    users = []
    for i in range(1, NUM_USERS + 1):
        city = random.choice(CITIES)
        segment = random.choices(['Premium', 'Budget'], weights=[0.3, 0.7])[0]
        order_frequency = np.random.poisson(5) + 1  # Average 5 orders
        avg_order_value = np.random.lognormal(3, 0.5) if segment == 'Premium' else np.random.lognormal(2.5, 0.5)
        users.append({
            'user_id': i,
            'city': city,
            'segment': segment,
            'order_frequency': order_frequency,
            'avg_order_value': avg_order_value
        })
    return pd.DataFrame(users)

def generate_restaurants():
    restaurants = []
    for i in range(1, NUM_RESTAURANTS + 1):
        cuisine = random.choice(CUISINES)
        rating = np.random.normal(4.0, 0.5)
        rating = max(1.0, min(5.0, rating))  # Clamp to 1-5
        restaurants.append({
            'restaurant_id': i,
            'cuisine': cuisine,
            'rating': round(rating, 1)
        })
    return pd.DataFrame(restaurants)

def generate_items():
    items = []
    for i in range(1, NUM_ITEMS + 1):
        category = random.choice(CATEGORIES)
        items.append({
            'item_id': i,
            'category': category
        })
    return pd.DataFrame(items)

def generate_orders(users_df, restaurants_df, items_df):
    orders = []
    for _ in range(NUM_ORDERS):
        user = users_df.sample(1).iloc[0]
        restaurant = restaurants_df.sample(1).iloc[0]
        # Simulate cart: 1-5 items
        cart_size = random.randint(1, 5)
        cart_items = items_df.sample(cart_size)

        # Time: hour_of_day
        hour = random.randint(8, 22)
        is_lunch = 11 <= hour <= 14
        is_dinner = 18 <= hour <= 21

        # Add-on logic: higher during lunch/dinner, premium more likely, dessert/drink after main
        has_main = 'Main' in cart_items['category'].values
        has_dessert_drink = any(cat in ['Dessert', 'Drink'] for cat in cart_items['category'].values)

        base_prob = 0.2
        if is_lunch or is_dinner:
            base_prob += 0.2
        if user['segment'] == 'Premium':
            base_prob += 0.15
        if has_main and not has_dessert_drink:
            base_prob += 0.1  # Suggest add-on

        addon_accepted = random.random() < base_prob

        # For each item in cart, but since target is per order? Wait, the target is addon_accepted per order I think.
        # The features are per order, target addon_accepted.

        orders.append({
            'user_id': user['user_id'],
            'city': user['city'],
            'order_frequency': user['order_frequency'],
            'avg_order_value': user['avg_order_value'],
            'segment': user['segment'],
            'restaurant_id': restaurant['restaurant_id'],
            'cuisine': restaurant['cuisine'],
            'rating': restaurant['rating'],
            'item_id': cart_items['item_id'].iloc[0],  # Representative item? Wait, perhaps need to adjust.
            # Actually, since cart_size, but for simplicity, pick one item, but category from cart.
            'category': random.choice(cart_items['category'].values),  # Random category from cart
            'hour_of_day': hour,
            'is_lunch': is_lunch,
            'is_dinner': is_dinner,
            'cart_size': cart_size,
            'addon_accepted': addon_accepted
        })

    return pd.DataFrame(orders)

if __name__ == "__main__":
    users_df = generate_users()
    restaurants_df = generate_restaurants()
    items_df = generate_items()
    orders_df = generate_orders(users_df, restaurants_df, items_df)

    # Save to CSV
    users_df.to_csv('data/users.csv', index=False)
    restaurants_df.to_csv('data/restaurants.csv', index=False)
    items_df.to_csv('data/items.csv', index=False)
    orders_df.to_csv('data/orders.csv', index=False)

    print("Synthetic data generated and saved to data/ folder.")
