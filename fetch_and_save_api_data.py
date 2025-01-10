import requests
import pandas as pd
import logging
import json
import subprocess
import sys

# API settings
API_URL_DATABASE = 'http://127.0.0.1:8000/api/admin/teachers-databases'
API_URL_TIME_CONFIG = 'http://127.0.0.1:8000/api/admin/times-configs'
BEARER_TOKEN = '4|amTbGO6oTdu2Ep7Yg5aJHrjXrQengH2R8cSyAh9ca3fb7c55'

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_teacher_data():
    """Fetch teacher data from the API."""
    headers = {
        'Authorization': f'Bearer {BEARER_TOKEN}'
    }

    try:
        response = requests.get(API_URL_DATABASE, headers=headers)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx, 5xx)
        logging.info("Data fetched successfully from teachers database")
        return response.json()['data']
    except requests.RequestException as e:
        logging.error(f"Failed to fetch data from teachers database: {e}")
        return []

def fetch_time_segment_data():
    """Fetch time segment data from the API."""
    headers = {
        'Authorization': f'Bearer {BEARER_TOKEN}'  # Menambahkan header Authorization
    }

    try:
        response = requests.get(API_URL_TIME_CONFIG, headers=headers)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx, 5xx)
        logging.info("Time segment data fetched successfully")
        return response.json()  # Mengambil data JSON dari respons
    except requests.RequestException as e:
        logging.error(f"Failed to fetch time segment data: {e}")
        return None



def save_to_csv(teacher_data):
    """Save teacher data to a CSV file."""
    if teacher_data:
        # Prepare the CSV data
        csv_data = []
        for teacher in teacher_data:
            record = {
                'id': teacher['id'],
                'position': teacher['position'],
                'face': teacher['face']
            }
            csv_data.append(record)

        # Create a DataFrame and save as CSV
        df = pd.DataFrame(csv_data)
        df.to_csv('data/teachers_data.csv', index=False)
        logging.info("Teacher data saved to teachers_data.csv")
    else:
        logging.warning("No teacher data to save")

def save_to_json(data, filename='config/time_segment.json'):
    """Save data to a JSON file."""
    if data:
        # Menyimpan data ke dalam file JSON
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        logging.info(f"Data saved to {filename}")
    else:
        logging.warning("No data to save")

def run_attendance_taker_script():
    """Run the new_attendance_taker.py script."""
    try:
        subprocess.run([sys.executable, "new_attandances_taker.py"], check=True)
        logging.info("Attendance taker script executed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running attendance taker script: {e}")

if __name__ == "__main__":
    # First attempt to fetch teacher data
    teacher_data = fetch_teacher_data()
    
    # Save teacher data to CSV
    save_to_csv(teacher_data)

    # Attempt to fetch time segment data
    time_segment_data = fetch_time_segment_data()

    if time_segment_data:
        # Save time segment data to JSON if successful
        save_to_json(time_segment_data)
        run_attendance_taker_script()
    else:
        # If time segment data fetch failed, skip and run the attendance taker script
        logging.info("Skipping time segment data fetch, running the attendance taker script.")
        run_attendance_taker_script()
