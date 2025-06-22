import cv2
import numpy as np
import face_recognition
import os
import sqlite3
from datetime import datetime

KNOWN_FACES_DIR = 'known_faces'
DB_NAME = 'attendance.db'
IMAGE_UPLOAD = 'class_photo.jpg'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS attendance (
                    name TEXT,
                    date TEXT,
                    time TEXT
                )""")
    conn.commit()
    conn.close()

def load_known_faces():
    known_encodings = []
    known_names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        img = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{filename}")
        encoding = face_recognition.face_encodings(img)
        if encoding:
            known_encodings.append(encoding[0])
            known_names.append(os.path.splitext(filename)[0])
    return known_encodings, known_names

def mark_attendance(name):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    c.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, date_str))
    if c.fetchone() is None:
        c.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)", (name, date_str, time_str))
    conn.commit()
    conn.close()

def process_class_image(image_path, known_encodings, known_names):
    img = cv2.imread(image_path)
    small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    rgb_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(known_encodings, encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_names[best_match_index]
            mark_attendance(name)
            print(f"Marked: {name}")
        else:
            print("Unknown face detected")

if __name__ == '__main__':
    init_db()
    known_encodings, known_names = load_known_faces()
    process_class_image(IMAGE_UPLOAD, known_encodings, known_names)
    print("Attendance marking complete.")
