import os
import dlib
import csv
import numpy as np
import logging
import cv2

# Path of cropped faces
path_images_from_camera = "data/data_faces_from_camera/"

# Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

def return_128d_features(path_img):
    img_rd = cv2.imread(path_img)
    faces = detector(img_rd, 1)
    logging.info("%-40s %-20s", " Image with faces detected:", path_img)

    # For photos of faces saved, we need to make sure that we can detect faces from the cropped images
    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = 0
        logging.warning("No face detected in image: %s", path_img)
    return face_descriptor

def return_features_mean_personX(path_face_personX):
    features_list_personX = []
    photos_list = os.listdir(path_face_personX)
    if photos_list:
        for i in range(len(photos_list)):
            logging.info("%-40s %-20s", " / Reading image:", path_face_personX + "/" + photos_list[i])
            features_128d = return_128d_features(path_face_personX + "/" + photos_list[i])
            if features_128d != 0:
                features_list_personX.append(features_128d)
    else:
        logging.warning("Warning: No images in %s/", path_face_personX)

    if features_list_personX:
        features_mean_personX = np.array(features_list_personX, dtype=object).mean(axis=0)
    else:
        features_mean_personX = np.zeros(128, dtype=object, order='C')
    return features_mean_personX

def read_existing_csv(csv_path):
    existing_data = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip the header
            for row in reader:
                if len(row) > 2:
                    existing_data[(row[0], row[1])] = row[2]  # Key by name and class
    return existing_data

def main():
    logging.basicConfig(level=logging.INFO)
    csv_path = "data/features_all.csv"
    person_list = os.listdir(path_images_from_camera)
    person_list.sort()

    # Read existing data
    existing_data = read_existing_csv(csv_path)

    # Jika file tidak ada atau kosong, tulis header
    file_exists = os.path.exists(csv_path)
    write_header = not file_exists or os.stat(csv_path).st_size == 0

    with open(csv_path, "a", newline="") as csvfile:  # Gunakan mode "a" untuk append
        writer = csv.writer(csvfile)

        # Tulis header jika file baru
        if write_header:
            writer.writerow(["name", "class", "face"])

        for person in person_list:
            logging.info("Processing: %s", person)
            features_mean_personX = return_features_mean_personX(path_images_from_camera + person)

            # Expected format: "person_1_name_class"
            parts = person.split('_')
            if len(parts) == 4:
                person_name = parts[2]  # e.g., "rival"
                person_class = parts[3]  # e.g., "guru"
            else:
                logging.warning("Unexpected filename format for person: %s", person)
                continue

            # Check if this entry already exists
            face_data = ",".join(map(str, features_mean_personX))  # Convert list of features to a single comma-separated string
            if (person_name, person_class) not in existing_data:
                writer.writerow([person_name, person_class, face_data])
                logging.info("Added new entry: %s", [person_name, person_class, face_data])
            else:
                logging.info("Entry already exists for: %s, %s", person_name, person_class)

        logging.info("Saved all the features of faces registered into: %s", csv_path)

if __name__ == '__main__':
    main()

