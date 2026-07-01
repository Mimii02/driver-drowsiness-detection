from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import pygame

pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")

def eye_aspect_ratio(eye):

    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20

counter = 0
alarm_on = False

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]    ####stores index numbers >start end positions of left eye in the shape predictor
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera not detected")
        break

    frame = imutils.resize(frame, width=600)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)                                       ###searches entire frame for faces.

    for face in faces:   ##process each face 

        shape = predictor(gray, face)   #finds 68 facial points 
        shape = face_utils.shape_to_np(shape)   ## convert lm to numpy array

        leftEye = shape[lStart:lEnd]      ##eye coordinates
        rightEye = shape[rStart:rEnd]   ##extracts only eye points 
        leftEAR = eye_aspect_ratio(leftEye)         ##measure how open left eye is
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0   #avg

        leftHull = cv2.convexHull(leftEye)      ##connect eye points create boundaries
        rightHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftHull], -1, (0, 255, 0), 1)    ##green c
        cv2.drawContours(frame, [rightHull], -1, (0, 255, 0), 1)

        if ear < EAR_THRESHOLD:

            counter += 1

            if counter >= CONSEC_FRAMES:

                cv2.putText(
                    frame,
                    "DROWSINESS ALERT!",
                    (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
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

        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    cv2.imshow("Driver Drowsiness Detection", frame)

    key = cv2.waitKey(1)

    if key == 27:  #### ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()