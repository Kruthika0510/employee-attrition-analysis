import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import zipfile

# Set random seed for reproducibility
np.random.seed(42)

@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    with zipfile.ZipFile('HRcommasep1603576336980.zip', 'r') as zip_ref:
        csv_filename = zip_ref.namelist()[0]
        with zip_ref.open(csv_filename) as file:
            df = pd.read_csv(file)
    
    # Standardize column names to lowercase
    df.columns = df.columns.str.lower()
    return df

def plot_satisfaction_hours(df):
    """Create satisfaction vs hours plot"""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df[df['left'] == 1], 
                   x='satisfaction_level', 
                   y='average_montly_hours',
                   alpha=0.5)
    plt.title('Satisfaction Level vs Working Hours (Employees who left)', fontsize=10)
    plt.xlabel('Satisfaction Level', fontsize=9)
    plt.ylabel('Average Monthly Hours', fontsize=9)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    return fig

def plot_department_analysis(df):
    """Create department-wise analysis plot"""
    dept_left = df[df['left'] == 1]['department'].value_counts()
    dept_total = df['department'].value_counts()
    dept_left_ratio = (dept_left / dept_total * 100).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    dept_left_ratio.plot(kind='bar')
    plt.title('Percentage of Employees Who Left by Department', fontsize=10)
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylabel('Percentage', fontsize=9)
    plt.tight_layout()
    return fig

def build_ml_model(df):
    """Build and evaluate the machine learning model"""
    # Prepare features
    categorical_columns = ['department', 'salary']
    le = LabelEncoder()
    df_encoded = df.copy()
    
    for col in categorical_columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    features = ['satisfaction_level', 'last_evaluation', 'number_project',
               'average_montly_hours', 'time_spend_company', 'work_accident',
               'promotion_last_5years', 'department', 'salary']
    
    X = df_encoded[features]
    y = df_encoded['left']
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Get predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return classification_report(y_test, y_pred), feature_importance

def main():
    st.set_page_config(page_title="Employee Attrition Analysis", layout="centered")
    
    st.title("Employee Attrition Analysis Dashboard")
    st.write("This dashboard analyzes employee attrition patterns and predicts potential exits.")
    
    # Load data
    df = load_data()
    
    # Data Overview in a smaller container
    st.header("Dataset Overview")
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Employees", f"{len(df):,}")
        with col2:
            st.metric("Employees Left", f"{df['left'].sum():,}")
        with col3:
            st.metric("Attrition Rate", f"{(df['left'].mean() * 100):.1f}%")
    
    # Satisfaction and Working Hours Analysis
    st.header("Satisfaction Level vs Working Hours")
    correlation = df[df['left'] == 1]['satisfaction_level'].corr(df[df['left'] == 1]['average_montly_hours'])
    st.write(f"Correlation: {correlation:.2f}")
    st.pyplot(plot_satisfaction_hours(df))
    
    # Department Analysis
    st.header("Department-wise Analysis")
    st.pyplot(plot_department_analysis(df))
    
    # Salary Analysis in columns
    st.header("Salary Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Salary Levels")
        st.write(df['salary'].value_counts().to_frame())
    
    with col2:
        st.subheader("Promotion Status")
        st.write(df['promotion_last_5years'].value_counts().to_frame())
    
    # Machine Learning Model Results
    st.header("Predictive Model Results")
    model_report, feature_importance = build_ml_model(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance")
        st.text(model_report)
    
    with col2:
        st.subheader("Feature Importance")
        st.dataframe(feature_importance, height=300)
    
    # Key Insights in an expandable section
    with st.expander("Key Insights", expanded=False):
        st.markdown("""
        1. **Satisfaction Level** is the most important factor in predicting employee exits
        2. **Time spent at company** and **Number of projects** are also significant factors
        3. Departments with highest attrition:
            - HR
            - Accounting
            - Technical
        4. Employees with low salaries have higher attrition rates
        """)

if __name__ == "__main__":
    main() 