import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
data = pd.read_csv('predictor/data.csv')

# Encode location
le = LabelEncoder()
data['location'] = le.fit_transform(data['location'])

X = data[['location', 'area', 'bedrooms', 'bathrooms']]
y = data['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model and encoder
joblib.dump(model, 'predictor/house_price_model.pkl')
joblib.dump(le, 'predictor/location_encoder.pkl')

print("Model trained and saved successfully")
