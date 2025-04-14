import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import zipfile

# Set random seed for reproducibility
np.random.seed(42)

def load_and_explore_data():
    """
    Load the dataset and perform initial exploration
    """
    # Extract and read the CSV file from zip
    with zipfile.ZipFile('HRcommasep1603576336980.zip', 'r') as zip_ref:
        csv_filename = zip_ref.namelist()[0]  # Get the first file in the zip
        with zip_ref.open(csv_filename) as file:
            df = pd.read_csv(file)
    
    # Standardize column names to lowercase
    df.columns = df.columns.str.lower()
    
    print("\nDataset Overview:")
    print("-" * 50)
    print("\nShape of the dataset:", df.shape)
    print("\nColumns in the dataset:", df.columns.tolist())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nData types:\n", df.dtypes)
    print("\nSummary statistics:\n", df.describe())
    
    return df

def analyze_satisfaction_hours(df):
    """
    Analyze relationship between satisfaction level and working hours
    """
    # Calculate correlation
    correlation = df[df['left'] == 1]['satisfaction_level'].corr(df[df['left'] == 1]['average_montly_hours'])
    print(f"\nCorrelation between satisfaction level and working hours: {correlation:.2f}")
    
    # Basic statistics for employees who left
    left_employees = df[df['left'] == 1]
    print("\nStatistics for employees who left:")
    print("\nSatisfaction level statistics:")
    print(left_employees['satisfaction_level'].describe())
    print("\nWorking hours statistics:")
    print(left_employees['average_montly_hours'].describe())

def analyze_multiple_factors(df):
    """
    Analyze the effect of multiple factors on employee attrition
    """
    # Department-wise analysis
    dept_left = df[df['left'] == 1]['department'].value_counts()
    dept_total = df['department'].value_counts()
    dept_left_ratio = (dept_left / dept_total * 100).sort_values(ascending=False)
    
    print("\nDepartment-wise attrition rates:")
    print(dept_left_ratio)
    
    # Promotion and salary analysis
    print("\nPromotion and salary analysis:")
    promotion_salary = pd.crosstab(df['promotion_last_5years'], df['salary'])
    print("\nCount of employees by promotion status and salary level:")
    print(promotion_salary)
    
    # Calculate percentage of employees who left by salary level
    salary_left = df[df['left'] == 1]['salary'].value_counts()
    salary_total = df['salary'].value_counts()
    salary_left_ratio = (salary_left / salary_total * 100).sort_values(ascending=False)
    
    print("\nPercentage of employees who left by salary level:")
    print(salary_left_ratio)

def build_ml_model(df):
    """
    Build and evaluate a machine learning model for predicting employee exit
    """
    # Prepare features
    categorical_columns = ['department', 'salary']
    le = LabelEncoder()
    df_encoded = df.copy()
    
    for col in categorical_columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    # Select features
    features = ['satisfaction_level', 'last_evaluation', 'number_project',
               'average_montly_hours', 'time_spend_company', 'work_accident',
               'promotion_last_5years', 'department', 'salary']
    
    X = df_encoded[features]
    y = df_encoded['left']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Print model evaluation metrics
    print("\nModel Evaluation:")
    print("-" * 50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return rf_model, feature_importance

def main():
    # Load and explore data
    df = load_and_explore_data()
    
    # Analyze satisfaction level and working hours
    analyze_satisfaction_hours(df)
    
    # Analyze multiple factors
    analyze_multiple_factors(df)
    
    # Build and evaluate ML model
    model, feature_importance = build_ml_model(df)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 