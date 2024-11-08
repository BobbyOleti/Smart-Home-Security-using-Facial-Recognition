from flask import Flask, render_template, Response
import cv2
import numpy as np
import pyttsx3
import os
import requests

app = Flask(__name__)

# Initialize the text-to-speech engine
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty("voice", voices[0].id)
engine.setProperty("rate", 140)
engine.setProperty("volume", 1000)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()
account_sid = 'AC9a8b1d825fd1980c90a3c80709f214e1'  # Replace with your Twilio Account SID
auth_token = '57c42e5ca4b2ef863250797526e47b10'  # Replace with your Twilio Auth Token
twilio_number = '+12407440852'  # Replace with your Twilio phone number in E.164 format

# Recipient's phone number (your own number)
my_number = '+918897439734'  # Replace with your own phone number in E.164 format

# API URL for sending SMS
api_url = f'https://api.twilio.com/2010-04-01/Accounts/AC9a8b1d825fd1980c90a3c80709f214e1/Messages.json'

def sendSMS(message):
    # Form data for the SMS
    data = {
        'To': my_number,
        'From': twilio_number,
        'Body': message,
    }

    try:
        # Sending the POST request to Twilio API
        response = requests.post(api_url, data=data, auth=(account_sid, auth_token))
        if response.status_code == 201:
            print('SMS sent successfully:', response.json())
            return True
        else:
            print('Failed to send SMS:', response.json())
            return False
    except requests.exceptions.RequestException as e:
        print('Error sending SMS:', e)
        return False

# Global variables
model = None
face_classifier = cv2.CascadeClassifier(r'C:\Users\AMALSIVAN\OneDrive\Desktop\ibmintern (2)\ibmintern\haarcascade_frontalface_default.xml')

# Function to load the trained model
def load_model():
    global model
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(r"C:\Users\AMALSIVAN\OneDrive\Desktop\ibmintern (2)\ibmintern\trained_model.yml")

# Function to extract face from an image
def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    for (x, y, w, h) in faces:
        cropped_face = img[y:y + h, x: x + w]
    return cropped_face

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to train the model (you need to implement this route separately)
@app.route('/train_model')
def train_model():
    # Placeholder for training model function (to be implemented separately)
    return "Model training complete and saved as trained_model.yml"

# Generator function for video frames
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Perform face recognition using the loaded model
            if model is not None:
                face = face_extractor(frame)
                if face is not None:
                    face = cv2.resize(face, (200, 200))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    result = model.predict(face)
                    if result[1] < 500:
                        confidence = int((1 - (result[1]) / 300) * 100)
                        display_string = str(confidence)
                        cv2.putText(frame, display_string, (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if confidence >= 80:  # Adjust confidence threshold as needed
                        cv2.putText(frame, "Unlocked", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    else:
                        cv2.putText(frame, "Locked", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        # Send SMS notification
                        message = "Unauthorized access detected! Face not recognized."
                        sendSMS(message)
                else:
                    cv2.putText(frame, "Face not found", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')

if __name__ == '__main__':
    load_model()  # Load the model when the application starts
    app.run(debug=True)