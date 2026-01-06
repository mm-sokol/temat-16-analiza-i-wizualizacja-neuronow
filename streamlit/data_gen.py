import numpy as np
import pandas as pd
from typing import Tuple

def generate_synthetic_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    # Features
    # Income: Normalized 0-1
    income = np.random.rand(n_samples)
    
    # Credit History: Normalized 0-1
    credit_history = np.random.rand(n_samples)
    
    # Age: 18-80
    age = np.random.randint(18, 81, n_samples)
    # Normalize Age for calculation
    age_norm = (age - 18) / (81 - 18)
    
    # Zip Code: 0 (Disadvantaged) or 1 (Advantaged)
    # This is our Protected Attribute
    zip_code = np.random.choice([0, 1], size=n_samples)
    
    # Random Noise
    noise = np.random.normal(0, 0.05, n_samples)
    
    # Target Calculation (Credit Score / Approval)
    # We inject bias: Zip Code has a strong positive weight
    # Income and Credit History also matter
    # Age has low weight
    weights = {
        'income': 0.5,
        'credit': 0.3,
        'age': 0.1,
        'zip': 1.5, # High weight for bias
        'noise': 1.0
    }
    
    linear_comb = (
        weights['income'] * income +
        weights['credit'] * credit_history +
        weights['age'] * age_norm +
        weights['zip'] * zip_code +
        weights['noise'] * noise
    )
    
    # Create a binary target based on a threshold
    # We want a balanced dataset roughly, or somewhat balanced
    threshold = np.mean(linear_comb)
    target = (linear_comb > threshold).astype(int)
    
    df = pd.DataFrame({
        'Income': income,
        'Credit History': credit_history,
        'Age': age,
        'Zip Code': zip_code,
        'Random Noise': noise,
        'Target': target
    })
    
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    print(df.head())
    print("\nCorrelations with Target:")
    print(df.corr()['Target'].sort_values(ascending=False))
