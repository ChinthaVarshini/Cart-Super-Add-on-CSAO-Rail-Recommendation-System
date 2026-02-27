import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def create_features(df):
    """
    Create features for the model.

    Args:
        df (pd.DataFrame): Orders dataframe.

    Returns:
        pd.DataFrame: Dataframe with engineered features.
    """
    # Time-based features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

    # User behavior features
    df['is_premium'] = (df['segment'] == 'Premium').astype(int)

    # Restaurant rating features
    df['rating_normalized'] = (df['rating'] - 1) / 4  # 0-1 scale

    # Cart context features
    df['has_main'] = (df['category'] == 'Main').astype(int)
    df['has_side'] = (df['category'] == 'Side').astype(int)
    df['has_dessert'] = (df['category'] == 'Dessert').astype(int)
    df['has_drink'] = (df['category'] == 'Drink').astype(int)

    # City encoding
    le_city = LabelEncoder()
    df['city_encoded'] = le_city.fit_transform(df['city'])

    # Cuisine encoding
    le_cuisine = LabelEncoder()
    df['cuisine_encoded'] = le_cuisine.fit_transform(df['cuisine'])

    # Category encoding
    le_category = LabelEncoder()
    df['category_encoded'] = le_category.fit_transform(df['category'])

    return df

def get_feature_store():
    """
    Simulate feature store for real-time inference.
    In production, this would be a database or feature store like Feast.

    Returns:
        dict: Mock feature store.
    """
    # For simplicity, return a dict. In real, query DB.
    return {
        'user_features': {},  # user_id -> features
        'restaurant_features': {},  # restaurant_id -> features
        'global_features': {}  # city, time, etc.
    }
