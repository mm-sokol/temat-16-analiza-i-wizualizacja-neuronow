import numpy as np
import pandas as pd
from typing import Tuple

def generate_synthetic_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    
    
    income = np.random.rand(n_samples)
    
    
    credit_history = np.random.rand(n_samples)
    
    
    age = np.random.randint(18, 81, n_samples)
    
    age_norm = (age - 18) / (81 - 18)
    
    
    
    zip_code = np.random.choice([0, 1], size=n_samples)
    
    
    noise = np.random.normal(0, 0.05, n_samples)
    
    
    
    
    
    weights = {
        'income': 0.5,
        'credit': 0.3,
        'age': 0.1,
        'zip': 1.5, 
        'noise': 1.0
    }
    
    linear_comb = (
        weights['income'] * income +
        weights['credit'] * credit_history +
        weights['age'] * age_norm +
        weights['zip'] * zip_code +
        weights['noise'] * noise
    )
    
    
    
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
