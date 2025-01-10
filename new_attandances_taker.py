import json
import dlib
import numpy as np
import cv2
import os
import pandas as pd
import datetime
import logging
import sqlite3
import base64
import requests
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

DATABASE = "attendance.db"
API_URL = "http://127.0.0.1:8000/api/admin/teachers-attandances"
BEARER_TOKEN = '4|amTbGO6oTdu2Ep7Yg5aJHrjXrQengH2R8cSyAh9ca3fb7c55'

class FaceRecognizer:
    def __init__(self, prayer_times):
        self.prayer_times = prayer_times
        self.face_features_known_list = []
        self.face_name_known_list = []
        self.load_known_faces()

    def calculate_distance(self, face1, face2):
        """Calculate the Euclidean distance between two face descriptors."""
        return np.linalg.norm(np.array(face1) - np.array(face2))

    def recognize_face(self, face_descriptor):
        """Recognize the face by comparing face descriptors."""
        min_distance = float('inf')
        match = None
        for i, known_descriptor in enumerate(self.face_features_known_list):
            distance = self.calculate_distance(face_descriptor, known_descriptor)
            if distance < 0.4 and distance < min_distance:
                min_distance = distance
                match = self.face_name_known_list[i]
        return match

    def load_known_faces(self):
        """Load known faces from the CSV file."""
        if os.path.exists("data/general_data.csv"):
            csv_rd = pd.read_csv("data/general_data.csv")
            for _, row in csv_rd.iterrows():
                teacher_id = row['id']
                position = row['position']
                features = [float(x) for x in row['face'].strip('[]').split(',')]
                self.face_name_known_list.append((teacher_id, position))
                self.face_features_known_list.append(features)
            logging.info("Loaded known faces: %d", len(self.face_features_known_list))
        else:
            logging.warning("teachers_data.csv not found. Please run fetch_and_save_api_data.py first.")

    def setup_database(self):
        """Initialize SQLite database for storing attendance data."""
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id_field TEXT,
                    position TEXT,
                    session_id INTEGER,
                    date TEXT,
                    time TEXT,
                    captured_image TEXT,
                    is_sent INTEGER DEFAULT 0
                )
            """)
            conn.commit()

    def save_to_database(self, user_id_field, position, session_id, date, time, image):
        """Save attendance data to SQLite database."""
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO attendance (user_id_field, position, session_id, date, time, captured_image, is_sent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id_field, position, session_id, date, time, image, 0))
            conn.commit()

    def send_to_api(self, record):
        """Send attendance data to API."""
        try:
            response = requests.post(API_URL, json=record, headers={'Authorization': f'Bearer {BEARER_TOKEN}'})
            return response.status_code == 200
        except requests.RequestException as e:
            logging.error("Failed to send data to API: %s", e)
        return False

    def retry_sending(self):
        """Retry sending unsent data with delay between retries."""
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM attendance WHERE is_sent = 0")
            unsent_records = cursor.fetchall()
            for record in unsent_records:
                data = {
                    "id": record[1],
                    "position": record[2],
                    "times_config_id": record[3],
                    "date": record[4],
                    "time": record[5],
                    "captured_image": record[6]
                }
                if self.send_to_api(data):
                    cursor.execute("UPDATE attendance SET is_sent = 1 WHERE id = ?", (record[0],))
                    logging.info("Data berhasil dikirim ke API")
                    time.sleep(2)  # Delay of 2 seconds between each data sent
            conn.commit()

    def record_attendance(self, user_id_field, position, session_id, timestamp, image):
        """Record attendance data and save it locally, check for duplicates."""
        current_date = timestamp.split(' ')[0]
        current_time = timestamp.split(' ')[1]

        # Cek apakah sudah ada data dengan user_id_field, session_id, dan date yang sama
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 1 FROM attendance WHERE user_id_field = ? AND session_id = ? AND date = ?
        """, (user_id_field, session_id, current_date))
        duplicate = cursor.fetchone()
        
        if duplicate:
            logging.info(f"[ABSEN] Data sudah ada untuk {user_id_field} pada sesi {session_id} tanggal {current_date}.")
            conn.close()
            return  # Tidak melakukan apa-apa jika data sudah ada

        # Convert image to Base64
        _, buffer = cv2.imencode('.jpg', image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        # Simpan data ke database
        self.save_to_database(user_id_field, position, session_id, current_date, current_time, encoded_image)
        logging.info("[ABSEN] %s (%s) berhasil absen pada %s %s", position, user_id_field, current_date, current_time)

        # Prepare data and convert non-serializable types
        data = {
            "id": str(user_id_field),  # Convert to string if it's an int64
            "position": position,
            "times_config_id": int(session_id),  # Ensure session_id is an int
            "date": current_date,
            "time": current_time,
            "captured_image": encoded_image
        }
        
        # Kirim data ke API jika belum ada
        if self.send_to_api(data):
            cursor.execute("UPDATE attendance SET is_sent = 1 WHERE user_id_field = ? AND date = ? AND time = ?", 
                        (user_id_field, current_date, current_time))
            conn.commit()
            logging.info(f"Data berhasil dikirim ke API untuk {user_id_field} pada {current_date} {current_time}")
        conn.close()

    def get_session(self, current_time):
        """Determine which prayer session the current time falls into."""
        current_time_obj = datetime.datetime.strptime(current_time, '%H:%M:%S').time()
        for segment in self.prayer_times:
            start = datetime.datetime.strptime(segment['start'], '%H:%M').time()
            end = datetime.datetime.strptime(segment['end'], '%H:%M').time()
            if (start <= current_time_obj <= end) or (start > end and (current_time_obj >= start or current_time_obj <= end)):
                return segment['id']
        return None

 
    def process_frame(self, frame):
        """Process each frame from the camera for face recognition."""
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        faces = detector(gray)
        
        # Menentukan padding untuk bounding box wajah
        padding = 20  # Bisa diatur untuk menambah ruang sekeliling wajah
        
        # Ambil waktu saat ini untuk sesi
        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        session = self.get_session(current_time)
        
        if session is None:
            return frame  # Jika tidak ada sesi yang cocok, kembalikan frame tanpa perubahan
        
        # Loop melalui semua wajah yang terdeteksi
        for face in faces:
            # Ambil koordinat wajah dan tambahkan padding
            top = max(0, face.top() - padding)
            left = max(0, face.left() - padding)
            bottom = face.bottom() + padding
            right = face.right() + padding
            
            # Gambar kotak biru dengan padding
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            
            # Ambil gambar wajah dengan padding (sekitar wajah)
            padded_face = frame[top:bottom, left:right]
            
            # Proses face descriptor untuk mengenali wajah
            shape = predictor(gray, face)
            rgb_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            face_descriptor = face_reco_model.compute_face_descriptor(rgb_image, shape)
            match = self.recognize_face(face_descriptor)
            
            position = None
            user_id_field = None
            
            if match:
                user_id_field, position = match
                
                # Record attendance
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.record_attendance(user_id_field, position, session, timestamp, frame)

        # Mengembalikan frame dengan kotak wajah yang digambar
        return frame




    def run(self):
        """Start the face recognition and attendance system."""
        self.setup_database()  # Pastikan tabel attendance sudah ada

        cap = cv2.VideoCapture(2)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.process_frame(frame)
            cv2.imshow("Face Recognizer", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        while True:  # Loop to retry sending data at the end time
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            session = self.get_session(current_time)
            if session is None:  # Only retry sending when end_time is reached
                self.retry_sending()
            time.sleep(2)  # Delay of 2 seconds between retries

if __name__ == "__main__":
    # Load prayer times from config/time_segment.json
    with open("config/time_segment.json", "r") as file:
        prayer_times = json.load(file)["time_segments"]

    recognizer = FaceRecognizer(prayer_times)
    recognizer.run()

