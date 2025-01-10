import json
import dlib
import numpy as np
import cv2
import os
import pandas as pd
import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

class FaceRecognizer:
    def __init__(self, prayer_times):
        self.prayer_times = prayer_times
        self.face_features_known_list = []
        self.face_name_known_list = []
        self.load_known_faces()
        self.absent_records = []  # List to track unique attendance

    def load_known_faces(self):
        if os.path.exists("data/features_all.csv"):
            csv_rd = pd.read_csv("data/features_all.csv")
            for i in range(csv_rd.shape[0]):
                self.face_name_known_list.append((csv_rd.iloc[i]['name'], csv_rd.iloc[i]['class']))  # (name, class)
                # Decode face column into a list of floats
                features = [float(x) for x in csv_rd.iloc[i]['face'].split(',') if x.strip()]
                self.face_features_known_list.append(features)
            logging.info("Loaded known faces: %d", len(self.face_features_known_list))
        else:
            logging.warning("features_all.csv not found. Run extraction script.")

    def get_session(self, current_time):
        current_time_obj = datetime.datetime.strptime(current_time, '%H:%M:%S').time()
        
        for segment in self.prayer_times:
            start = datetime.datetime.strptime(segment['start'], '%H:%M').time()
            end = datetime.datetime.strptime(segment['end'], '%H:%M').time()
            
            # Check time range including cases crossing midnight
            if start <= end:
                if start <= current_time_obj <= end:
                    return segment['name']
            else:
                if current_time_obj >= start or current_time_obj <= end:
                    return segment['name']
        
        return None

    def record_attendance(self, name, class_name, session, timestamp):
        current_date = timestamp.split(' ')[0]

        # Read data from CSV file to avoid duplicates
        filename = f"attendance_{current_date}.csv"
        if os.path.exists(filename):
            existing_records = pd.read_csv(filename).to_dict('records')
        else:
            existing_records = []

        # Create a new record
        record = {
            "name": name,
            "class": class_name,
            "session": session,
            "date": current_date,
            "time": timestamp.split(' ')[1]
        }

        # Check if the record already exists in CSV or memory
        if not any(existing_record['name'] == record['name'] and
                   existing_record['class'] == record['class'] and
                   existing_record['session'] == record['session'] and
                   existing_record['date'] == record['date']
                   for existing_record in (self.absent_records + existing_records)):
            self.absent_records.append(record)
            self.save_attendance(record)
            logging.info("[ABSEN] %s - %s (%s) berhasil absen at %s", name, class_name, session, timestamp)
        else:
            logging.info("[SUDAH ABSEN] %s - %s (%s) sudah absen at %s", name, class_name, session, timestamp)

    def save_attendance(self, record):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        filename = f"attendance_{current_date}.csv"
        
        df = pd.DataFrame([record])
        
        if not os.path.exists(filename):
            df.to_csv(filename, mode='w', header=True, index=False)
        else:
            df.to_csv(filename, mode='a', header=False, index=False)

    @staticmethod
    def calculate_distance(face1, face2):
        return np.linalg.norm(np.array(face1) - np.array(face2))

    def recognize_face(self, face_descriptor):
        min_distance = float('inf')
        match = None
        for i, known_descriptor in enumerate(self.face_features_known_list):
            distance = self.calculate_distance(face_descriptor, known_descriptor)
            if distance < 0.4 and distance < min_distance:
                min_distance = distance
                match = self.face_name_known_list[i]
        return match

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        session = self.get_session(current_time)

        if session:
            for face in faces:
                shape = predictor(gray, face)
                rgb_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                face_descriptor = face_reco_model.compute_face_descriptor(rgb_image, shape)
                match = self.recognize_face(face_descriptor)
                if match:
                    name, class_name = match
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self.record_attendance(name, class_name, session, timestamp)
                    cv2.putText(frame, f"{name}, {class_name}, {session}", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        return frame

    def run(self):
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

if __name__ == "__main__":
    # Load prayer times from config/time_segment.json
    with open("config/time_segment.json", "r") as file:
        config_data = json.load(file)
        prayer_times = config_data["time_segments"]

    recognizer = FaceRecognizer(prayer_times)
    recognizer.run()
