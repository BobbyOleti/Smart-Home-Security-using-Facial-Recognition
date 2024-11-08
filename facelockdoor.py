import cv2
import pyttsx3
import numpy as np
from os import listdir
from os.path import isfile, join
import serial
import time

# Initialize serial communication with Arduino
arduino = serial.Serial('COM11', 9600)  # Replace 'COM3' with your Arduino port

# Initialize the text-to-speech engine
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty("voice", voices[0].id)
engine.setProperty("rate", 140)
engine.setProperty("volume", 1.0)  # Set volume between 0.0 and 1.0

def speak(audio):
    """ Function to convert text to speech """
    print(f"Speaking: {audio}")  # Debug print
    engine.say(audio)
    engine.runAndWait()

# Path to training data and confidence threshold for recognition
DATA_PATH = r"C:\Users\AMALSIVAN\OneDrive\Desktop\ibmintern (2)\ibmintern\Face1"
CONFIDENCE_THRESHOLD = 83  

# Load Haar Cascade for face detection
face_classifier = cv2.CascadeClassifier(r'C:\Users\AMALSIVAN\OneDrive\Desktop\ibmintern (2)\ibmintern\haarcascade_frontalface_default.xml')

def face_extractor(img):
    """ Function to detect and extract face from an image """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    for (x, y, w, h) in faces:
        cropped_face = img[y:y + h, x: x + w]
    return cropped_face

# Load training data and labels
onlyfiles = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))]
Training_data, Labels = [], []

for i, file in enumerate(onlyfiles):
    image_path = join(DATA_PATH, file)
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if images is not None:
        images = cv2.resize(images, (200, 200))
        Training_data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)
    else:
        print(f"Warning: Could not read image file {image_path}")

Labels = np.asarray(Labels, dtype=np.int32)

# Check if training data is loaded properly
if len(Training_data) == 0:
    print("No training data found. Exiting...")
    exit(1)

# Train the LBPH face recognizer model
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_data), np.asarray(Labels))
print("Training complete")

def face_detector(img):
    """ Function to detect faces and draw rectangle around them """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return img, None
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y + h, x: x + w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi

# Initialize video capture
cap = cv2.VideoCapture(0)

x = 0  # Counter for recognized faces
c = 0  # Counter for unrecognized faces
d = 0  # Counter for face not found
m = 0  # Motor control flag
face_not_found_counter = 0  # To avoid repeated 'Face not found' messages

messages = []  # List to store messages for feedback

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    image, face = face_detector(frame)

    if face is not None:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)
        if result[1] < 500:
            confidence = int((1 - (result[1]) / 300) * 100)
            display_string = str(confidence)
            cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if confidence >= CONFIDENCE_THRESHOLD:
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow('Face Recognition', image)
            arduino.write(b'1')  # Send signal to Arduino to turn on the motor
            x += 1
            face_not_found_counter = 0  # Reset the 'face not found' counter
        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Recognition', image)
            arduino.write(b'0')  # Send signal to Arduino to turn off the motor
            c += 1
            face_not_found_counter = 0  # Reset the 'face not found' counter
    else:
        if face_not_found_counter == 0:  # Avoid spamming the error message
            cv2.putText(image, "Face not found", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            arduino.write(b'0')  # Ensure motor is off if no face is detected
        cv2.imshow('Face Recognition', image)
        face_not_found_counter += 1
        d += 1

    if cv2.waitKey(1) == 13 or x == 10 or c == 30 or d == 20:
        break

cap.release()
cv2.destroyAllWindows()

# Final action based on counters
if x >= 5:
    m = 1
    messages.append("Welcome home Boss")
    messages.append("Good to have you back")
    arduino.write(b'1')  # Send signal to Arduino to turn on the motor
    time.sleep(3)  # Wait for 3 seconds before turning off the motor
    arduino.write(b'0')  # Send signal to Arduino to turn off the motor
elif c == 30:
    messages.append("Face not recognized. Please try again.")
    arduino.write(b'0')  # Ensure motor is off
elif d == 20:
    messages.append("Face not found. Please try again.")
    arduino.write(b'0')  # Ensure motor is off

# Speak and display final messages
for message in messages:
    speak(message)
    print(message)
