# streamlit_app.py

import cv2
import os
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

model = load_model(os.path.join("models", "model_test.h5"))
lbl = ['Close', 'Open']

class VideoProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, minNeighbors=3, scaleFactor=1.1, minSize=(25, 25))
        eyes = eye_cascade.detectMultiScale(gray, minNeighbors=1, scaleFactor=1.1)

        cv2.rectangle(frm, (0, frm.shape[0] - 50), (200, frm.shape[0]), (0, 0, 0), thickness=cv2.FILLED)

        for (x, y, w, h) in faces:
            cv2.rectangle(frm, (x, y), (x + w, y + h), (255, 0, 0), 3)

        for (x, y, w, h) in eyes:
            eye = frm[y:y + h, x:x + w]
            eye = cv2.resize(eye, (80, 80))
            eye = eye / 255
            eye = eye.reshape(80, 80, 3)
            eye = np.expand_dims(eye, axis=0)
            prediction = model.predict(eye)

            if prediction[0][0] > 0.30:
                cv2.putText(frm, "Closed", (10, frm.shape[0] - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (255, 255, 255), 1, cv2.LINE_AA)
            elif prediction[0][1] > 0.70:
                cv2.putText(frm, "Open", (10, frm.shape[0] - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1,
                            cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(frm, format='bgr24')

# Streamlit app
def main():
    st.title("Drowsiness Detection with Webcam")

    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}], "audio": False}
    )

    webrtc_streamer(
        key="example",
        video_processor_factory=VideoProcessor,
        rtc_configuration=rtc_configuration,
    )

# Run the Streamlit app
if __name__ == '__main__':
    main()
