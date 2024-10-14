import requests
import pandas as pd
import numpy as np
from numpy.polynomial import Polynomial
from datetime import datetime
import json
import matplotlib.pyplot as plt

# API setup
BASE_URL = "https://api.rapt.io"
GET_HYDROMETERS_URL = f"{BASE_URL}/api/Hydrometers/GetHydrometers"
GET_TELEMETRY_URL = f"{BASE_URL}/api/Hydrometers/GetTelemetry"
AUTH_URL = "https://id.rapt.io/connect/token"

# Function to get bearer token
def get_bearer_token(username, api_key):
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "client_id": "rapt-user",
        "grant_type": "password",
        "username": username,
        "password": api_key
    }
    response = requests.post(AUTH_URL, headers=headers, data=data)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        raise Exception(f"Error obtaining bearer token: {response.status_code}")

# Function to get hydrometers
def get_hydrometers(headers):
    response = requests.get(GET_HYDROMETERS_URL, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error getting hydrometers: {response.status_code}")

# Function to get telemetry data for a specific hydrometer
def get_telemetry(hydrometer_id, start_date, end_date, headers):
    params = {
        "hydrometerId": hydrometer_id,
        "startDate": start_date,
        "endDate": end_date
    }
    response = requests.get(GET_TELEMETRY_URL, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error getting telemetry data: {response.status_code}")

# Perform predictions using polynomial model
def perform_prediction(telemetry_data, prediction_hours=1, prediction_periods=168, degree=3, predict_history=24):
    df = pd.DataFrame(telemetry_data)
    df["createdOn"] = pd.to_datetime(df["createdOn"])
    df = df.sort_values("createdOn")

    # Converting timestamps to numeric values for regression
    df["timestamp"] = (df["createdOn"] - df["createdOn"].min()).dt.total_seconds() / 3600  # Time in hours from start
    X_full = df["timestamp"].values
    y_full = df["gravity"].values
    temperature = df["temperature"].values

    # Determine Original Gravity (OG)
    original_gravity = y_full.max()

    # Filter to only include the last predict_history hours of data for prediction
    predict_timestamps = df["createdOn"].max() - pd.to_timedelta(predict_history, unit='h')
    predict_dataset = df[df["createdOn"] >= predict_timestamps]
    X = predict_dataset["timestamp"].values
    y = predict_dataset["gravity"].values

    if len(X) < 2 or len(y) < 2:
        raise ValueError("Not enough data points for prediction.")

    # Fit a polynomial model to the gravity data from the last 24 hours
    p_gravity = Polynomial.fit(X, y, degree)

    # Predict future gravity trends
    future_timestamps = np.array([X[-1] + prediction_hours * i for i in range(1, prediction_periods + 1)])
    future_predictions_gravity = p_gravity(future_timestamps)

    # Determine when SG levels off (using threshold for minimal change)
    leveling_off_index = np.argmax(np.diff(future_predictions_gravity) > 0)
    found_level_off = future_predictions_gravity[leveling_off_index]
    if found_level_off:
        future_timestamps = future_timestamps[:leveling_off_index + 1]
        future_predictions_gravity = future_predictions_gravity[:leveling_off_index + 1]
        leveling_off_date = df["createdOn"].max() + pd.to_timedelta(leveling_off_index * prediction_hours, unit='h')
        leveling_off_value = future_predictions_gravity[-1]
    else:
        leveling_off_date = "Not found"
        leveling_off_value = None

    # Predict future gravity trends up to final prediction point
    final_timestamp = future_timestamps[-1] if found_level_off else future_timestamps[-1]
    full_range_timestamps = np.linspace(X_full.min(), final_timestamp, num=1000)
    full_range_predictions_gravity = p_gravity(full_range_timestamps)

    # Calculate ABV
    if leveling_off_value is not None:
        final_gravity = leveling_off_value
        abv = 132.715 * ((original_gravity - final_gravity) / 1000)
    else:
        final_gravity = None
        abv = None

    # Plotting the data
    plt.figure(figsize=(12, 9))
    plt.subplot(3, 1, 1)
    plt.plot(df["createdOn"], y_full, label="Observed Gravity", marker='o')
    plt.plot(pd.to_datetime(df["createdOn"].min()) + pd.to_timedelta(full_range_timestamps, unit='h'), full_range_predictions_gravity, label="Gravity Prediction Up to Final", linestyle='--', color='orange')
    if found_level_off:
        plt.axvline(x=leveling_off_date, color='red', linestyle=':', label=f'Predicted Final Gravity ({leveling_off_date.strftime("%Y-%m-%d %H:%M")})')
        plt.text(leveling_off_date, leveling_off_value, f'{leveling_off_value:.2f}', color='red', fontsize=10, verticalalignment='bottom')
    plt.axhline(y=original_gravity, color='blue', linestyle='-.', label=f'Original Gravity ({original_gravity:.2f})')
    plt.xlabel("Date")
    plt.ylabel("Gravity")
    plt.title("Observed and Predicted Gravity Over Time (Polynomial Model with Predicted Final Gravity)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(df["createdOn"], temperature, label="Observed Temperature", marker='o', color='green')
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.title("Observed Temperature Over Time")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    if final_gravity is not None:
        abv_values = 132.715 * ((original_gravity - full_range_predictions_gravity) / 1000)
        plt.plot(pd.to_datetime(df["createdOn"].min()) + pd.to_timedelta(full_range_timestamps, unit='h'), abv_values, label="Calculated ABV", linestyle='-', color='purple')
        plt.axhline(y=abv, color='red', linestyle=':', label=f'Final ABV ({abv:.2f}%)')
    plt.xlabel("Date")
    plt.ylabel("ABV (%)")
    plt.title("Calculated Alcohol by Volume (ABV) Over Time")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return future_predictions_gravity, leveling_off_date

# Main script
def main():
    try:
        # Load user credentials from JSON file
        with open('credentials.json', 'r') as f:
            credentials = json.load(f)

        username = credentials['username']
        api_key = credentials['api_key']

        # Get bearer token
        token = get_bearer_token(username, api_key)
        headers = {
            "Authorization": f"Bearer {token}"
        }

        # Specify the date range
        start_date = "2024-10-11T00:00:00Z"
        end_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Get available devices
        devices = get_hydrometers(headers)
        for device in devices:
            print(f"Downloading telemetry for hydrometer: {device['name']} ({device['id']})")
            telemetry_data = get_telemetry(device['id'], start_date, end_date, headers)

            if telemetry_data:
                future_predictions_gravity, leveling_off_date = perform_prediction(telemetry_data, prediction_hours=1, prediction_periods=168, degree=2, predict_history=24)
                print(f"Predicted leveling off date for SG: {leveling_off_date}")
            else:
                print("No telemetry data available for this device.")

    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    main()