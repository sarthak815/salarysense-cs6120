# SalrySense: Salary Prediction Machine Learning Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0.2-orange.svg)](https://scikit-learn.org/stable/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.6.2-green.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3.2-yellow.svg)](https://lightgbm.readthedocs.io/)

## Overview

This project implements a comprehensive machine learning pipeline for predicting salary ranges from job postings. Using advanced NLP techniques and ensemble learning methods, we achieve highly accurate salary predictions based on job descriptions, titles, required skills, and location data.

The repository contains two Jupyter notebooks analyzing different job posting datasets:

1. `salary_prediction_linkedin.ipynb` - Analyzes LinkedIn job postings data
2. `salary_prediction_glassdoor.ipynb` - Analyzes Glassdoor job postings data

## Key Features

- **Text Processing**: Advanced TF-IDF vectorization of job descriptions and titles
- **Feature Engineering**: 
  - Extraction of years of experience requirements
  - Education level detection
  - Seniority classification from job titles
  - Geographic region encoding and tech hub indicators
- **Model Evaluation**: Comprehensive metrics including adjusted MAE, within-10% accuracy, and R²
- **Error Analysis**: Detailed examination of extreme prediction errors
- **Calibration Analysis**: Bias-variance analysis across salary ranges
- **Visualization**: Extensive plotting of model performance and feature importance

## Dataset-Specific Techniques

### LinkedIn Analysis
- **Keyword Extraction**: Custom regex patterns to extract years of experience (e.g., "5+ years experience")
- **Education Requirements**: Pattern matching to detect education level requirements (Bachelor's, Master's, PhD)
- **Seniority Classification**: Rule-based classification from job titles (Junior, Senior, Manager, Director, etc.)
- **Location Processing**: Conversion of locations to geographic regions and tech hub indicators
- **Skill Identification**: Boolean features for technical skills like Python, R, AWS, etc.

### Glassdoor Analysis
- **Company Age**: Normalized company age extracted and used as a predictive feature
- **Job Title Vectorization**: Fine-tuned TF-IDF specifically for the shorter Glassdoor job titles
- **Industry Mapping**: Extraction and encoding of industry information from job descriptions
- **Compensation Components**: Analysis of different compensation types (base, bonus, stock options)
- **Employment Type**: Categorical encoding of full-time, contract, and part-time positions

## Detailed Results

### Model Performance Comparison

| Model | Adjusted MAE ($) | Within 10% (%) | R² |
|-------|------------------|----------------|------|
| XGBoost (n=100, d=6, lr=0.1) | 7,404.48 ± 1,173.61 | 70.79 ± 2.80 | 0.7244 ± 0.0654 |
| LightGBM (n=100, lvs=63, lr=0.1) | 7,512.36 ± 1,076.48 | 68.75 ± 3.42 | 0.7163 ± 0.0657 |
| LightGBM (n=100, lvs=31, lr=0.1) | 7,512.36 ± 1,076.48 | 68.75 ± 3.42 | 0.7163 ± 0.0657 |
| Ridge Regression (α=0.1) | 8,121.37 ± 955.18 | 63.45 ± 1.75 | 0.6920 ± 0.0543 |
| Random Forest (n=100, d=20) | 8,463.96 ± 1,145.45 | 51.63 ± 3.63 | 0.6857 ± 0.0665 |

### LinkedIn vs. Glassdoor Results

**LinkedIn Dataset:**
- Higher accuracy for technical roles (tech industry jobs)
- Better performance in the $100K-$200K range
- Years of experience and education level features showed significant predictive power
- Tech hub indicators were highly significant predictive features

**Glassdoor Dataset:**
- Better performance for non-technical roles
- More accurate in the lower salary ranges ($50K-$100K)
- Company age emerged as an important feature
- Employment type (full-time vs. contract) showed high predictive value

### Error Analysis

Our best model (XGBoost) showed the following error characteristics:

- Mean absolute error: $7,404.48
- 70.79% of predictions within 10% of actual value
- Error distribution:
  - Under-predictions: 70.3% of extreme errors
  - Over-predictions: 29.7% of extreme errors
- Salary range error distribution:
  - $50K-$100K: 24.3% of extreme errors
  - $100K-$150K: 8.1% of extreme errors
  - $150K-$200K: 21.6% of extreme errors
  - >$200K: 45.9% of extreme errors

### Calibration Analysis

The model shows excellent calibration in the $80K-$150K range, with increasing bias at the extremes:

- Slight over-prediction for salaries below $80K (positive bias)
- High accuracy in middle ranges ($80K-$150K)
- Under-prediction for very high salaries above $200K (negative bias)

## Models Evaluated

### Linear Regression
- **Implementation**: Scikit-learn's LinearRegression
- **Performance**: Poor performance with extreme overfitting
- **Limitations**: Unable to capture non-linear relationships in salary data

### Ridge Regression
- **Implementation**: Scikit-learn's Ridge with α regularization
- **Hyperparameters Tested**: α ∈ {0.1, 1.0, 10.0, 100.0}
- **Best Configuration**: α=0.1
- **Performance**: Good baseline with Adjusted MAE of $8,121.37
- **Benefit**: Effective regularization preventing overfitting

### Random Forest
- **Implementation**: Scikit-learn's RandomForestRegressor
- **Hyperparameters Tested**: 
  - n_estimators ∈ {50, 100}
  - max_depth ∈ {10, 20}
- **Best Configuration**: n_estimators=100, max_depth=20
- **Performance**: Decent with Adjusted MAE of $8,463.96
- **Feature Importance**: Provided valuable insights on important features

### XGBoost
- **Implementation**: XGBoost's XGBRegressor
- **Hyperparameters Tested**:
  - n_estimators = 100
  - max_depth ∈ {3, 6}
  - learning_rate ∈ {0.01, 0.1}
- **Best Configuration**: n_estimators=100, max_depth=6, learning_rate=0.1
- **Performance**: Best overall performance with Adjusted MAE of $7,404.48
- **Advantages**: Excellent handling of complex feature interactions

### LightGBM
- **Implementation**: LightGBM's LGBMRegressor
- **Hyperparameters Tested**:
  - n_estimators = 100
  - num_leaves ∈ {31, 63}
  - learning_rate ∈ {0.01, 0.1}
- **Best Configuration**: n_estimators=100, num_leaves=31 or 63, learning_rate=0.1
- **Performance**: Second best with Adjusted MAE of $7,512.36
- **Advantages**: Fast training time while maintaining accuracy

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- matplotlib
- seaborn
- nltk

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/salary-prediction-project.git
cd salary-prediction-project
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the notebooks
```bash
jupyter notebook
```
Then open either `salary_prediction_linkedin.ipynb` or `salary_prediction_glassdoor.ipynb`

## Keyword Extraction Techniques

The LinkedIn analysis incorporates several specialized keyword extraction techniques:

### Years of Experience Extraction
```python
def extract_years_experience(text):
    """Extract years of experience requirement from job description"""
    patterns = [
        r'(\d+)\+?\s*years?(?:\s*of)?\s*experience',
        r'(\d+)-(\d+)\s*years?(?:\s*of)?\s*experience',
        r'experience:\s*(\d+)\+?\s*years?',
        r'experience(?:\s*of)?\s*(\d+)\+?\s*years?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            # Handle ranges like "3-5 years"
            if len(match.groups()) > 1 and match.group(2):
                return (int(match.group(1)) + int(match.group(2))) / 2
            return int(match.group(1))
    
    return 0  # Default to 0 if no match found
```

### Education Level Detection
```python
def extract_education_level(text):
    """Extract education level requirement from job description"""
    text = text.lower()
    
    # Education level scoring (higher number = higher education)
    if re.search(r'phd|doctorate|doctoral', text):
        return 4
    elif re.search(r'master\'?s|ms degree|m.s.|mba|m.b.a', text):
        return 3
    elif re.search(r'bachelor\'?s|bachelors|bs degree|b.s.|ba degree|b.a.', text):
        return 2
    elif re.search(r'associate\'?s|associates|community college', text):
        return 1
    else:
        return 0
```

### Seniority Classification
```python
def extract_seniority(title):
    """Extract seniority level from job title"""
    title = title.lower()
    
    # Seniority scoring (higher = more senior)
    if re.search(r'chief|ceo|cto|cfo|coo|president|vp|vice president', title):
        return 5
    elif re.search(r'director|head', title):
        return 4
    elif re.search(r'senior|sr|lead', title):
        return 3
    elif re.search(r'manager|supervisor', title):
        return 2
    elif re.search(r'junior|jr|associate|intern|assistant', title):
        return 1
    else:
        return 2  # Default to mid-level if no indicator
```

## Dataset Structure

The datasets include:
- Job titles and descriptions (text data)
- Boolean features for required skills (Python, R, Spark, AWS, Excel)
- Company age
- Location data (transformed into tech hub indicators and geographic regions)
- Salary information (minimum, maximum, and average)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Future Work

- Web application deployment for real-time salary prediction
- API integration with job posting platforms
- Implementation of deep learning approaches (BERT, RoBERTa for NLP)
- Time series analysis to capture salary trends over time
- Exploration of additional features like company size and industry
- Cross-industry validation and transfer learning

## Acknowledgements

- LinkedIn and Glassdoor for the job posting data
- The scikit-learn, XGBoost, and LightGBM communities for their excellent tools
- All contributors who have provided feedback and suggestions
