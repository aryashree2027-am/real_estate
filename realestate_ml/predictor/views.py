from django.shortcuts import render
import joblib
import numpy as np

model = joblib.load('predictor/house_price_model.pkl')
encoder = joblib.load('predictor/location_encoder.pkl')

def home(request):
    predicted_price = None

    if request.method == 'POST':
        location = request.POST['location']
        area = int(request.POST['area'])
        bedrooms = int(request.POST['bedrooms'])
        bathrooms = int(request.POST['bathrooms'])

        location_encoded = encoder.transform([location])[0]

        features = np.array([[location_encoded, area, bedrooms, bathrooms]])
        predicted_price = model.predict(features)[0]

    return render(request, 'home.html', {'price': predicted_price})

