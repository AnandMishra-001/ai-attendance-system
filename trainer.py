# trainer.py
# Phase 2: Generate face encodings from collected images

import face_recognition
import os
import pickle

def train_faces(dataset_path="dataset", encoding_file="encodings.pkl"):
    """
    Loops through all images in dataset/<StudentName>/
    Generates 128-d face encodings for each image
    Saves all encodings + names to encodings.pkl
    """

    known_encodings = []   # list of 128-d vectors
    known_names = []       # list of matching student names

    # ── 1. Loop through each student folder ──────────────────────────
    students = os.listdir(dataset_path)
    print(f"[INFO] Found {len(students)} student(s): {students}")

    for student_name in students:
        student_folder = os.path.join(dataset_path, student_name)

        if not os.path.isdir(student_folder):
            continue   # skip any stray files

        images = os.listdir(student_folder)
        print(f"[INFO] Processing '{student_name}' — {len(images)} images...")

        # ── 2. Process each image ─────────────────────────────────────
        for img_file in images:
            img_path = os.path.join(student_folder, img_file)

            # Load image in RGB format (face_recognition needs RGB, not BGR)
            image = face_recognition.load_image_file(img_path)

            # Generate encoding(s) — returns a list (usually 1 face per image)
            # model="hog" is faster; model="cnn" is more accurate (needs GPU)
            encodings = face_recognition.face_encodings(image, model="hog")

            if len(encodings) == 0:
                # No face found in this image — skip it
                print(f"  [SKIP] No face detected in {img_file}")
                continue

            # Take the first (and usually only) encoding
            known_encodings.append(encodings[0])
            known_names.append(student_name)

    # ── 3. Save encodings to disk ─────────────────────────────────────
    data = {
        "encodings": known_encodings,
        "names": known_names
    }

    with open(encoding_file, "wb") as f:
        pickle.dump(data, f)

    print(f"\n[DONE] Trained {len(known_encodings)} face(s) from {len(students)} student(s).")
    print(f"[DONE] Encodings saved to '{encoding_file}'")


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    train_faces()