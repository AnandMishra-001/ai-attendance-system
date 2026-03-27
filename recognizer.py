# recognizer.py
# Phase 3 & 4: Live face recognition + attendance logging

import face_recognition
import cv2
import pickle
import csv
import os
from datetime import datetime

# ── 1. Load saved encodings ───────────────────────────────────────────
print("[INFO] Loading encodings...")
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]
print(f"[INFO] Loaded {len(known_encodings)} encodings.")

# ── 2. Setup attendance CSV ───────────────────────────────────────────
csv_file = "attendance.csv"

# Write header row if file doesn't exist yet
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

# Track who has already been marked this session (avoid duplicates)
marked_today = set()

def mark_attendance(name):
    """Append a row to attendance.csv if not already marked this session."""
    if name in marked_today:
        return  # already logged, skip

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, date_str, time_str])

    marked_today.add(name)
    print(f"[ATTENDANCE] Marked: {name} at {time_str}")

# ── 3. Start webcam ───────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("[INFO] Starting live recognition. Press 'q' to quit.")

# ── 4. Live recognition loop ──────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to 1/4 size for faster processing
    # (we scale coordinates back up for display)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # face_recognition needs RGB; OpenCV gives BGR
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and compute encodings in this frame
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for (face_encoding, face_loc) in zip(face_encodings, face_locations):

        # Compare this face against all known encodings
        # Returns a list of True/False for each known encoding
        matches = face_recognition.compare_faces(
            known_encodings,
            face_encoding,
            tolerance=0.5   # lower = stricter. 0.5 is a good default
        )

        name = "Unknown"

        # Use the known face with the smallest distance (most similar)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        if len(face_distances) > 0:
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_names[best_match_index]
                mark_attendance(name)   # log to CSV

        # Scale face location back up (we shrank the frame by 4x earlier)
        top, right, bottom, left = face_loc
        top *= 4; right *= 4; bottom *= 4; left *= 4

        # Color: green for known, red for unknown
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        # Draw rectangle and name label
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

    # Show marked count on screen
    cv2.putText(frame, f"Marked today: {len(marked_today)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Attendance System — Press 'q' to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── 5. Cleanup ────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print(f"\n[DONE] Session ended. Total marked: {len(marked_today)}")
print(f"[DONE] Attendance saved to '{csv_file}'")