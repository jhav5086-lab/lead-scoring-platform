import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

def create_sample_models():
    """Create sample model files for testing the Streamlit app"""
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Sample feature columns (based on your app's expected features)
    feature_columns = [
        'age', 'duration', 'campaign', 'pdays', 'previous', 
        'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
        'euribor3m', 'nr.employed', 'engagement_score', 
        'digital_intent_score', 'product_count',
        'job_encoded', 'marital_encoded', 'education_encoded',
        'default_encoded', 'housing_encoded', 'loan_encoded',
        'contact_encoded', 'month_encoded', 'day_of_week_encoded',
        'poutcome_encoded'
    ]
    
    # Create sample label encoders
    label_encoders = {}
    
    categorical_columns = {
        'job': ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
                'retired', 'self-employed', 'services', 'student', 'technician', 
                'unemployed', 'unknown'],
        'marital': ['married', 'single', 'divorced', 'unknown'],
        'education': ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
                     'professional.course', 'university.degree', 'unknown'],
        'default': ['no', 'yes', 'unknown'],
        'housing': ['no', 'yes', 'unknown'],
        'loan': ['no', 'yes', 'unknown'],
        'contact': ['cellular', 'telephone'],
        'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 
                 'oct', 'nov', 'dec'],
        'day_of_week': ['mon', 'tue', 'wed', 'thu', 'fri'],
        'poutcome': ['nonexistent', 'failure', 'success']
    }
    
    for col, categories in categorical_columns.items():
        le = LabelEncoder()
        le.fit(categories)
        label_encoders[col] = le
    
    # Create a sample Random Forest model
    # Using dummy data for initialization
    X_dummy = np.random.randn(100, len(feature_columns))
    y_dummy = np.random.randint(0, 2, 100)
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf_model.fit(X_dummy, y_dummy)
    
    # Save the models
    with open('models/rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    with open('models/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    
    print("✅ Sample model files created successfully!")
    print("📁 Files saved in 'models/' directory:")
    print("   - rf_model.pkl")
    print("   - label_encoders.pkl") 
    print("   - feature_columns.pkl")

if __name__ == "__main__":
    create_sample_models()
