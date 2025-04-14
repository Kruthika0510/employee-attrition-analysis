# Employee Attrition Analysis

This project analyzes employee attrition data to understand the factors contributing to employee exits and builds a machine learning model to predict potential exits.

## Project Overview

The analysis includes:
1. Data exploration and preprocessing
2. Analysis of satisfaction level vs. working hours
3. Multi-factor analysis (department, promotion, salary)
4. Machine learning model for predicting employee exits

## Setup and Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Ensure the HR dataset (HRcommasep1603576336980.zip) is in the root directory

3. Run the analysis:
```bash
python employee_attrition_analysis.py
```

## Output

The script generates several visualization files:
- satisfaction_hours_relationship.png: Scatter plot showing relationship between satisfaction and working hours
- department_analysis.png: Bar chart showing department-wise attrition rates
- promotion_salary_analysis.png: Analysis of promotion and salary distribution
- feature_importance.png: Feature importance plot from the machine learning model

## Model Details

The project uses a Random Forest Classifier to predict employee exits. The model:
- Processes both numerical and categorical features
- Performs feature scaling
- Provides feature importance analysis
- Generates a detailed classification report 