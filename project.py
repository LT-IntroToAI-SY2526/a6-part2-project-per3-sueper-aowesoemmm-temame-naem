"""
Multivariable Linear Regression Project
Assignment 6 Part 3: Lane Tech College Prep

Group Members:
- Noah Kim
- Nicky Chiang
- Maddox Bartoli

Dataset: Adult Census Income
Predicting: Income Likelihood (>50K)
Features: Age, Educational-Num, Hours-per-Week
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# CONFIGURATION
DATA_FILE = 'income_train.csv'
TARGET = 'income_>50K'  # Corrected capitalization
FEATURES = ['age', 'educational-num', 'hours-per-week']

def load_and_explore_data(filename):
    """Load data, clean missing values, and print summary stats."""
    print("=" * 70)
    print("STEP 1: LOADING AND EXPLORING DATA")
    print("=" * 70)
    
    df = pd.read_csv(filename)
    
    # Cleaning: Remove rows with missing values (the 5767 you found)
    df = df.dropna()
    
    print(f"Dataset Loaded: {df.shape[0]} rows and {df.shape[1]} columns.")
    print("\nSummary Statistics for Features:")
    print(df[FEATURES].describe())
    
    return df

def visualize_data(df):
    """Create scatter plots to show relationships for the presentation."""
    print("\n" + "=" * 70)
    print("STEP 2: VISUALIZING RELATIONSHIPS")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('How Features Impact Income Likelihood', fontsize=16)

    for i, col in enumerate(FEATURES):
        # Using alpha=0.1 because there are many data points
        axes[i].scatter(df[col], df[TARGET], alpha=0.1, color='royalblue')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(TARGET)
        axes[i].set_title(f'{col} vs Income')

    plt.tight_layout()
    plt.savefig('income_visualizations.png')
    print("✓ Plots saved as 'income_visualizations.png' for your slides.")
    plt.show()

def train_and_evaluate(df):
    """Split data, train the model, and calculate performance metrics."""
    print("\n" + "=" * 70)
    print("STEP 3 & 4: MODEL TRAINING & EVALUATION")
    print("=" * 70)

    # Prepare X and y
    X = df[FEATURES]
    y = df[TARGET]

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 1. Coefficients (For Presentation Slide 3)
    print("\n--- Model Coefficients ---")
    for name, coef in zip(FEATURES, model.coef_):
        print(f"{name}: {coef:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")

    # 2. Performance Metrics (For Presentation Slide 4)
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"\n--- Performance Results ---")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # 3. Comparison Table (For Presentation Slide 4)
    comparison = pd.DataFrame({'Actual': y_test, 'Predicted': predictions.round(2)})
    print("\nExample Comparisons (Actual vs Predicted):")
    print(comparison.head(10))

    return model

def make_custom_prediction(model):
    """Example prediction for the presentation demo."""
    print("\n" + "=" * 70)
    print("STEP 5: EXAMPLE PREDICTION")
    print("=" * 70)
    
    # Example: Person is 45 years old, 16 years education (Masters/Doc), 50 hours/week
    person = pd.DataFrame([[45, 16, 50]], columns=FEATURES)
    result = model.predict(person)[0]
    
    print("Input: Age=45, Education=16, Hours=50")
    print(f"Predicted Likelihood of earning >50K: {result:.2%}")

if __name__ == "__main__":
    data = load_and_explore_data(DATA_FILE)
    visualize_data(data)
    trained_model = train_and_evaluate(data)
    make_custom_prediction(trained_model)
    
    print("\n" + "=" * 70)
    print("PROJECT COMPLETE - GOOD LUCK ON THE FINALS!")
    print("=" * 70)