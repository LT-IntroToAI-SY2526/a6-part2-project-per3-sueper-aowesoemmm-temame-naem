
"""
Multivariable Linear Regression Project
Assignment 6 Part 3

Group Members:
- Noah Kim
- Nicky Chiang
- Maddox Bartoli
- 

Dataset: [Name of your dataset]
Predicting: [What you're predicting]
Features: [List your features]
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# TODO: Update this with your actual filename
DATA_FILE = 'income_train.csv'

def load_and_explore_data(filename):
    """
    Load your dataset and print basic information
    
    TODO:
    - Load the CSV file
    - Print the shape (rows, columns)
    - Print the first few rows
    - Print summary statistics
    - Check for missing values
    """
    print("=" * 70)
    print("LOADING AND EXPLORING DATA")
    print("=" * 70)
    
    # Your code here
    
    data = pd.read_csv(filename)
    # TODO: Print the first 5 rows
    print("=== Job Income Data ===")
    print(f"\nFirst 5 rows:")
    print(data.head())
    # TODO: Print the shape of the dataset
    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
    # TODO: Print basic statistics for ALL columns
    print(f"\nBasic statistics:")
    print(data.describe())
    # TODO: Print the column names
    print(f"\nColumn names: {list(data.columns)}")
    # TODO: Return the dataframe
    return data


def visualize_data(data):
    """
    Create visualizations to understand your data
    
    TODO:
    - Create scatter plots for each feature vs target
    - Save the figure
    - Identify which features look most important
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
    """
    print("\n" + "=" * 70)
    print("VISUALIZING RELATIONSHIPS")
    print("=" * 70)
    
    # Your code here
    # Hint: Use subplots like in Part 2!

    fig, axes = plt.subplots(2, 3, figsize=(12, 10))
    fig.suptitle('Income Features vs Income', fontsize=16, fontweight='bold')
    
    # Plot 1: age vs income
    axes[0, 0].scatter(data['age'], data['income_>50k'], color='blue', alpha=0.6)
    axes[0, 0].set_xlabel('age (years)')
    axes[0, 0].set_ylabel('income_>50k (0=below, 1=above)')
    axes[0, 0].set_title('age vs income_>50k')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: educational-num vs income
    axes[0, 1].scatter(data['educational-num'], data['income_>50k'], color='green', alpha=0.6)
    axes[0, 1].set_xlabel('educational-num (years)')
    axes[0, 1].set_ylabel('income_>50k (0=below, 1=above)')
    axes[0, 1].set_title('educational-num vs income_>50k')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: relationship vs income
    axes[1, 0].scatter(data['relationship'], data['income_>50k'], color='red', alpha=0.6)
    axes[1, 0].set_xlabel('relationship (0=Other, 1=Husband, 2=Not-in-family)')
    axes[1, 0].set_ylabel('income_>50k (0=below, 1=above)')
    axes[1, 0].set_title('relationship vs income_>50k')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 3: race vs income
    axes[1, 1].scatter(data['race'], data['income_>50k'], color='brown', alpha=0.6)
    axes[1, 1].set_xlabel('race (0=Other, 1=Black, 2=White)')
    axes[1, 1].set_ylabel('income_>50k (0=below, 1=above)')
    axes[1, 1].set_title('race vs income_>50k')
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 3: gender vs income
    axes[0, 2].scatter(data['gender'], data['income_>50k'], color='purple', alpha=0.6)
    axes[0, 2].set_xlabel('gender (0=male, 1=female)')
    axes[0, 2].set_ylabel('income_>50k (0=below, 1=above)')
    axes[0, 2].set_title('gender vs income_>50k')
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 3: hours-per-week vs Price
    axes[1, 2].scatter(data['hours-per-week'], data['income_>50k'], color='black', alpha=0.6)
    axes[1, 2].set_xlabel('hours-per-week (hours)')
    axes[1, 2].set_ylabel('income_>50k (0=below, 1=above)')
    axes[1, 2].set_title('hours-per-week vs income_>50k')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('income_features.png', dpi=300, bbox_inches='tight')
    print("\n✓ Feature plots saved as 'income_features.png'")
    plt.show()

def prepare_and_split_data(data):
    """
    Prepare X and y, then split into train/test
    
    TODO:
    - Separate features (X) and target (y)
    - Split into train/test (80/20)
    - Print the sizes
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 70)
    print("PREPARING AND SPLITTING DATA")
    print("=" * 70)
    
    # Your code here
    
    pass


def train_model(X_train, y_train):
    """
    Train the linear regression model
    
    TODO:
    - Create and train a LinearRegression model
    - Print the equation with all coefficients
    - Print feature importance (rank features by coefficient magnitude)
    
    Args:
        X_train: training features
        y_train: training target
        feature_names: list of feature names
        
    Returns:
        trained model
    """
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    
    # Your code here
    
    pass


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    TODO:
    - Make predictions on test set
    - Calculate R² score
    - Calculate RMSE
    - Print results clearly
    - Create a comparison table (first 10 examples)
    
    Args:
        model: trained model
        X_test: test features
        y_test: test target
        
    Returns:
        predictions
    """
    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)
    
    # Your code here
    
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    print(f"\n=== Model Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"  → Model explains {r2*100:.2f}% of income variation")
    
    print(f"\nRoot Mean Squared Error: ${rmse:.2f}")
    print(f"  → On average, predictions are off by ${rmse:.2f}")
    
    print(f"\n=== Feature Importance ===")
    feature_importance = list(zip(feature_names, np.abs(model.coef_)))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, importance) in enumerate(feature_importance, 1):
        print(f"{i}. {name}: {importance:.2f}")
    
    return predictions


def make_prediction(model):
    """
    Make a prediction for a new example
    
    TODO:
    - Create a sample input (you choose the values!)
    - Make a prediction
    - Print the input values and predicted output
    
    Args:
        model: trained model
        feature_names: list of feature names
    """
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTION")
    print("=" * 70)
    
    # Your code here
    # Example: If predicting house price with [sqft, bedrooms, bathrooms]
    # sample = pd.DataFrame([[2000, 3, 2]], columns=feature_names)
    
    income_variables = pd.DataFrame([age, educational-num, relationship, race, gender, hours-per-week], columns=['Age', 'Education', 'Relationship', 'Race', 'Gender', 'Hours per Week'])
    predicted_income = model.predict(income_variables)[0]

    print(f"\n=== New Prediction ===")
    print(f"Income Variables: {age:.0f}years, {education}Level of Education, {relationship} Relationship, {race} Race,{gender} Gender, {hours-per-week}Hours Worked Per Week")
    print(f"Predicted Income: ${predicted_income:,.2f}")
    
    return predicted_income


if __name__ == "__main__":
    # Step 1: Load and explore
    data = load_and_explore_data(DATA_FILE)
    
    # Step 2: Visualize
    visualize_data(data)
    
    # Step 3: Prepare and split
    X_train, X_test, y_train, y_test = prepare_and_split_data(data)
    
    # Step 4: Train
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate
    predictions = evaluate_model(model, X_test, y_test)
    
    # Step 6: Make a prediction, add features as an argument
    make_prediction(model)
    
    print("\n" + "=" * 70)
    print("PROJECT COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Analyze your results")
    print("2. Try improving your model (add/remove features)")
    print("3. Create your presentation")
    print("4. Practice presenting with your group!")
