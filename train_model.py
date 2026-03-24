import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Sample dataset
data = {
    "area": [1000, 1500, 2000, 2500],
    "bedrooms": [2, 3, 3, 4],
    "age": [5, 10, 3, 8],
    "price": [3000000, 4500000, 6000000, 8000000]
}

df = pd.DataFrame(data)

X = df[["area", "bedrooms", "age"]]
y = df["price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model saved as model.pkl")