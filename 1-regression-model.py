```python
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data = {
    "experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "salary": [30000, 35000, 45000, 50000, 60000,
               65000, 75000, 80000, 90000, 100000]
}

df = pd.DataFrame(data)

# Features (Input)
X = df[["experience"]]

# Target (Output)
y = df["salary"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

print("Actual Salary:")
print(y_test.values)

print("\nPredicted Salary:")
print(y_pred)

# Error calculation
mae = mean_absolute_error(y_test, y_pred)
print(f"\nMean Absolute Error: {mae:.2f}")

# Predict new value
salary_prediction = model.predict([[11]])

print(f"\nPredicted Salary for 11 years experience: £{salary_prediction[0]:.2f}")
```
