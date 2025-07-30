import cv2
import dlib
import time
import math
from scipy.spatial import distance
import numpy as np


def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def calculate_MAR(mouth):
    A = distance.euclidean(mouth[2], mouth[10])  # Vertical
    B = distance.euclidean(mouth[4], mouth[8])   # Vertical
    C = distance.euclidean(mouth[0], mouth[6])   # Horizontal
    return (A + B) / (2.0 * C)


def get_head_pose(shape):
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),     # Nose tip
        (shape.part(8).x, shape.part(8).y),       # Chin
        (shape.part(36).x, shape.part(36).y),     # Left eye left corner
        (shape.part(45).x, shape.part(45).y),     # Right eye right corner
        (shape.part(48).x, shape.part(48).y),     # Left Mouth corner
        (shape.part(54).x, shape.part(54).y)      # Right mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1)
    ])

    focal_length = 1
    center = (0, 0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pitch = math.degrees(math.asin(-rotation_mat[2][0]))
    return pitch

cap = cv2.VideoCapture("http://192.168.165.118:8080/video")

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("/home/cheesecake/Downloads/shape_predictor_68_face_landmarks.dat")

# Thresholds
EAR_THRESHOLD = 0.26
MAR_THRESHOLD = 0.55
CLOSED_EYES_TIME = 2
YAWNING_TIME = 2
PITCH_THRESHOLD = 15  # Degrees
STILLNESS_TIME = 7  # Seconds

# Timers
start_eye_time = None
start_yawn_time = None
start_tilt_time = None
still_start_time = None

prev_nose_pos = None
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from phone camera stream")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)

    eye_detected = yawn_detected = tilt_detected = still_detected = False

    for face in faces:
        landmarks = dlib_facelandmark(gray, face)

        leftEye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        rightEye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]
        mouth = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)]

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)
        EAR = round((left_ear + right_ear) / 2, 2)

        MAR = round(calculate_MAR(mouth), 2)
        # EAR check
        if EAR < EAR_THRESHOLD:
            if start_eye_time is None:
                start_eye_time = time.time()
            elif time.time() - start_eye_time >= CLOSED_EYES_TIME:
                eye_detected = True
                cv2.putText(frame, "Drowsiness: Eyes Closed", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            start_eye_time = None
        if MAR > MAR_THRESHOLD:
            if start_yawn_time is None:
                start_yawn_time = time.time()
            elif time.time() - start_yawn_time >= YAWNING_TIME:
                yawn_detected = True
                cv2.putText(frame, "Drowsiness: Yawning", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            start_yawn_time = None
        pitch = get_head_pose(landmarks)
        if abs(pitch) > PITCH_THRESHOLD:
            if start_tilt_time is None:
                start_tilt_time = time.time()
            elif time.time() - start_tilt_time >= 2:
                tilt_detected = True
                cv2.putText(frame, "Drowsiness: Head Tilt", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            start_tilt_time = None
        current_nose = (landmarks.part(30).x, landmarks.part(30).y)
        if prev_nose_pos is not None:
            movement = distance.euclidean(current_nose, prev_nose_pos)
            if movement < 10:  # barely moved
                if still_start_time is None:
                    still_start_time = time.time()
                elif time.time() - still_start_time >= STILLNESS_TIME:
                    still_detected = True
                    cv2.putText(frame, "Drowsiness: No Movement", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            else:
                still_start_time = None
        prev_nose_pos = current_nose

        print(f"EAR: {EAR}, MAR: {MAR}, Pitch: {round(pitch, 2)}")

    if eye_detected or yawn_detected or tilt_detected or still_detected:
        print("Drowsiness signs detected!")

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()