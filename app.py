# from flask import Flask
# app = Flask(__name__)

# @app.route("/")
# def home():
#     return "HELLOOOOOWWWWW"

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# @app.route("/")

# Load the saved LOF model based on the sensor name
def load_model(sensor_name):
    model_path = os.path.join(os.path.dirname(__file__), 'model', f'model_{sensor_name}_lof.pkl')
    return joblib.load(model_path)
    
# Function to preprocess the new data for a single sensor
def preprocess_sensor_data(sensor_data):
    # Handle missing values by imputing with median
    imputer = SimpleImputer(strategy='median')
    data_imputed = imputer.fit_transform(sensor_data)

    # Preprocess with RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(data_imputed)

    # Apply PCA
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca

@app.route('/detect_anomalies', methods=['POST'])
def detect_anomalies():
    data = request.json
    sensor_name = data['sensor_name']
    sensor_data = pd.DataFrame(data['sensor_data'])
    
    # Preprocess the data for the sensor
    X_pca = preprocess_sensor_data(sensor_data)
    
    # Load the corresponding model
    lof_model = load_model(sensor_name)
    
    # Predict anomalies using the loaded model
    y_pred = lof_model.fit_predict(X_pca)
    y_pred_binary = np.where(y_pred == -1, 1, 0)
    
    return jsonify({'anomalies': y_pred_binary.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
