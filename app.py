import streamlit as st
import cv2
import dlib
import imutils
import pygame
import time
from scipy.spatial import distance
from imutils import face_utils

st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")

st.title(" AI Based Driver Drowsiness Detection System")

st.markdown("""
- Romaisa Mariam 
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")

# EAR Function
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Thresholds
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20

counter = 0
alarm_on = False

# Load dlib
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Dashboard Layout
col1, col2 = st.columns([3, 1])

video_placeholder = col1.empty()

ear_placeholder = col2.empty()
status_placeholder = col2.empty()
score_placeholder = col2.empty()
timer_placeholder = col2.empty()

# Camera
start_btn = st.button("Start Detection")

if not start_btn:
    st.stop()
cap = cv2.VideoCapture(0)

start_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        st.error("Camera not detected")
        break

    frame = imutils.resize(frame, width=700)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftHull = cv2.convexHull(leftEye)
        rightHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftHull], -1, (0,255,0), 1)
        cv2.drawContours(frame, [rightHull], -1, (0,255,0), 1)

        if ear < EAR_THRESHOLD:

            counter += 1

            if counter >= CONSEC_FRAMES:

                cv2.putText(
                    frame,
                    "DROWSINESS ALERT!",
                    (100,100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3
                )

                if not alarm_on:
                    pygame.mixer.music.play(-1)
                    alarm_on = True

        else:

            counter = 0

            if alarm_on:
                pygame.mixer.music.stop()
                alarm_on = False

        status = "NORMAL"

        if counter >= CONSEC_FRAMES:
            status = "DROWSY"

        elapsed_time = int(time.time() - start_time)

        ear_placeholder.metric("EAR", f"{ear:.2f}")
        status_placeholder.metric("Status", status)
        score_placeholder.metric("Score", counter)
        timer_placeholder.metric("Timer", f"{elapsed_time}s")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    video_placeholder.image(
        frame,
        channels="RGB",
        use_container_width=True
    )

cap.release()
