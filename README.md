# Employee Attrition Analysis

This project analyzes employee attrition data to understand the factors contributing to employee exits and builds a machine learning model to predict potential exits.

## Project Overview

The analysis includes:
1. Data exploration and preprocessing
2. Analysis of satisfaction level vs. working hours
3. Multi-factor analysis (department, promotion, salary)
4. Machine learning model for predicting employee exits

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/Kruthika0510/employee-attrition-analysis.git
cd employee-attrition-analysis
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
python -m streamlit run app.py
```

4. Open the app in your web browser:
- The app will be available at http://localhost:8501

## Output

The dashboard includes:
- Interactive visualizations
- Dataset overview
- Satisfaction vs working hours analysis
- Department-wise attrition rates
- Salary and promotion analysis
- Machine learning model results
- Key insights and recommendations

## Model Details

The project uses a Random Forest Classifier to predict employee exits. The model:
- Processes both numerical and categorical features
- Performs feature scaling
- Provides feature importance analysis
- Generates a detailed classification report 