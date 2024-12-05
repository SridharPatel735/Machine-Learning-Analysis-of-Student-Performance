# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
data = pd.read_csv('Student_performance_data _.csv', header=None, sep=',')
data.columns = ['ID', 'AGE', 'GENDER', 'ETHNICITY', 'PARENT_EDU', 'STUDYTIME', 'ABSENCES',
                'TUTORING', 'PARENT_SUP', 'EXTRACURRICULAR', 'SPORTS', 'MUSIC',
                'VOLUNTEERING', 'GPA', 'GRADECLASS']

# Drop the 'ID' column as it is not needed for analysis
data = data.drop('ID', axis=1)

# Display dataset preview and basic info
print("Dataset Preview:")
print(data.head())

print("\nDataset Info:")
print(data.info())

# Check for missing values in the dataset
print("\nMissing Values:")
print(data.isnull().sum())

# Exploratory Data Analysis
print("\nDataset Statistics:")
print(data.describe())

# Visualize pairplot of features to understand relationships
sns.pairplot(data)
plt.suptitle("Pairplot of Features", y=1.02)
plt.show()

# Correlation heatmap to visualize relationships between features
nan_counts = data.isna().sum()
cols_to_use = [col for col in data.columns if nan_counts[col] < len(data)]
correlation_matrix = np.corrcoef(data[cols_to_use].values.T)

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cbar=True, annot=True, square=True, fmt='.2f',
            annot_kws={'size': 8}, yticklabels=cols_to_use, xticklabels=cols_to_use)
plt.title("Correlation Heatmap of Features")
plt.show()

# Data Preprocessing
# Encode categorical variables using LabelEncoder
categorical_cols = data.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Handle missing values by filling them with the mean of the column
data.fillna(data.mean(), inplace=True)

# Define feature matrix X and target variable y
X = data.drop('GPA', axis=1)
y = data['GPA']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize regression models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
    "Random Forest Regressor": RandomForestRegressor(random_state=42, n_estimators=100),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
}

# Train and evaluate each model with cross-validation
results = []
cv_results = []

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')  # 5-fold CV, using R² as the scoring metric
    avg_cv_score = np.mean(cv_scores)

    # Store results
    cv_results.append({"Model": name, "CV R² (Mean)": avg_cv_score})
    results.append({"Model": name, "RMSE": rmse, "R² Score": r2, "Explained Variance Score": evs})

    # Print model performance
    print(f"\n{name}:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"Explained Variance Score: {evs:.2f}")
    print(f"Cross-Validation Mean R² Score: {avg_cv_score:.2f}")

# Convert results to DataFrame for easy viewing
results_df = pd.DataFrame(results)
cv_results_df = pd.DataFrame(cv_results)

# Display model performance comparison
print("\nModel Performance Comparison:")
print(results_df)

print("\nCross-Validation Performance Comparison:")
print(cv_results_df)

# Visualization of model performance (R² Score)
plt.figure(figsize=(10, 6))
sns.barplot(x='R² Score', y='Model', data=results_df.sort_values(by='R² Score', ascending=False))
plt.title("Model Performance Comparison (R² Score)")
plt.xlabel("R² Score")
plt.ylabel("Model")
plt.show()

# Visualization of cross-validation performance
plt.figure(figsize=(10, 6))
sns.barplot(x='CV R² (Mean)', y='Model', data=cv_results_df.sort_values(by='CV R² (Mean)', ascending=False))
plt.title("Cross-Validation Performance Comparison (Mean R² Score)")
plt.xlabel("Mean R² Score (Cross-Validation)")
plt.ylabel("Model")
plt.show()

# Feature importance visualization for Random Forest
importances = models["Random Forest Regressor"].feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importance (Random Forest)")
plt.show()