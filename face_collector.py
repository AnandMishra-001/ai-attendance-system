# face_collector.py
# Phase 1: Collect face images for each student

import cv2
import os

def collect_faces(student_name, num_images=100):
    """
    Opens the webcam, detects the student's face using a Haar cascade,
    and saves 'num_images' cropped face images to dataset/<student_name>/
    """

    # ── 1. Setup save folder ──────────────────────────────────────────
    save_path = os.path.join("dataset", student_name)
    os.makedirs(save_path, exist_ok=True)   # creates folder if not exists

    # ── 2. Load the face detector ─────────────────────────────────────
    # OpenCV ships this XML file — it detects frontal faces using Haar features
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # ── 3. Open webcam ────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)   # 0 = default/built-in webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    count = 0   # tracks how many images we've saved

    print(f"[INFO] Collecting {num_images} images for '{student_name}'")
    print("[INFO] Press 'q' to quit early.")

    # ── 4. Frame loop ─────────────────────────────────────────────────
    while count < num_images:
        ret, frame = cap.read()     # ret = True if frame captured successfully
        if not ret:
            print("Error: Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        # Haar cascade works on grayscale images

        # Detect faces in the frame
        # scaleFactor: how much image size is reduced at each scale
        # minNeighbors: how many neighbors each rectangle should have to be retained
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(80, 80)    # ignore very small detections
        )

        for (x, y, w, h) in faces:
            # Crop the detected face region from the original color frame
            face_crop = frame[y:y+h, x:x+w]

            # Resize to a standard size (makes training more consistent)
            face_resized = cv2.resize(face_crop, (160, 160))

            # Save image as dataset/StudentName/img_0.jpg, img_1.jpg, ...
            img_path = os.path.join(save_path, f"img_{count}.jpg")
            cv2.imwrite(img_path, face_resized)
            count += 1

            # Draw a green rectangle around the detected face (live preview)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Show image count on screen
            cv2.putText(frame, f"Saved: {count}/{num_images}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

            if count >= num_images:
                break   # stop once we hit the target

        # Display the live webcam feed with the green rectangle
        cv2.imshow("Face Collector — Press 'q' to quit", frame)

        # Exit early if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ── 5. Cleanup ────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Saved {count} images to '{save_path}'")


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    name = input("Enter student name (no spaces, e.g. Rahul_Sharma): ").strip()
    collect_faces(name, num_images=100)