from flask import Flask, request, jsonify, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingRegressor
from fuzzywuzzy import process
import pandas as pd
from flask_cors import CORS  # Import CORS
import os

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"},
                    r"/recommendation": {"origins": "http://localhost:3000"}})

# Path to frontend build directory
frontend_build_dir = '../frontend/build'

# Load the doctor dataset from CSV
doctors_df = pd.read_csv('../backend/doctors.csv')  # Adjust the path as needed

# Load the symptoms and specialist dataset from CSV
symptoms_specialist_df = pd.read_csv('../backend/symptoms_specialist.csv')  # Adjust the path as needed

# Load the trained model and prepare data
data = pd.read_csv("../backend/data.csv")  # Adjust the path as needed
features = data.drop(['Total'], axis='columns')
target = data["Total"]
new_features = pd.get_dummies(features, drop_first=True)

# Train the model
model = BaggingRegressor()
model.fit(new_features, target)

@app.route('/')
def serve_frontend():
    return send_from_directory(frontend_build_dir, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(frontend_build_dir, filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_responses = [int(request.json[f'col{i}']) for i in range(10)]
        input_data = pd.DataFrame([user_responses], columns=new_features.columns)
        predicted_total = model.predict(input_data)[0]
        percent = (predicted_total / 300) * 100
        print('Predicted Total:', predicted_total)
        return jsonify({'predicted_total': percent})
    except Exception as e:
        print('Error predicting result:', str(e))
        return jsonify({'error': str(e)})

@app.route('/recommendation', methods=['POST'])
def get_recommendation():
    data = request.json
    selected_symptom = data['selectedSymptom']
    matched_symptom = process.extractOne(selected_symptom, symptoms_specialist_df['symptoms'])
    if matched_symptom[1] >= 80:
        filtered_specialist = symptoms_specialist_df[symptoms_specialist_df['symptoms'] == matched_symptom[0]]
    else:
        filtered_specialist = pd.DataFrame()

    recommended_doctors = []
    if not filtered_specialist.empty:
        if len(filtered_specialist['specialist'].unique()) == 1:
            predicted_specialist = filtered_specialist['specialist'].unique()[0]
            filtered_doctors = doctors_df[doctors_df['Specialty'] == predicted_specialist]
        else:
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(filtered_specialist['symptoms'])
            y = filtered_specialist['specialist']
            classifier = LinearSVC()
            classifier.fit(X, y)

            X_user = vectorizer.transform([selected_symptom])
            predicted_specialist = classifier.predict(X_user)

            filtered_doctors = doctors_df[doctors_df['Specialty'] == predicted_specialist[0]]

        for index, row in filtered_doctors.iterrows():
            recommended_doctors.append({
                'Name': row['Name'],
                'Specialty': row['Specialty'],
                'Location': row['Location'],
                'Experience': row['Experience'],
                'Contact': row['Contact'],
                'Profile Picture': row['Profile Picture']
            })

    return jsonify(recommended_doctors)

if __name__ == '__main__':
    app.run(debug=True)
