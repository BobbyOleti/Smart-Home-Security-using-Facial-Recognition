import cv2
import pyttsx3
import numpy as np
import os

# Initialize the face classifier
face_classifier = cv2.CascadeClassifier(r'C:\Users\AMALSIVAN\OneDrive\Desktop\ibmintern (2)\ibmintern\haarcascade_frontalface_default.xml')


# Initialize the text-to-speech engine
def speak(audio):
    engine.say(audio)
    engine.runAndWait()

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty("voice", voices[0].id)
engine.setProperty("rate", 140)
engine.setProperty("volume", 1000)

xcount = 0

# Function to load the count of existing images
def load_count():
    try:
        with open('count.txt', 'r') as file:
            count = int(file.read().strip())
    except FileNotFoundError:
        count = 0
    return count

# Function to save the count of images
def save_count(count):
    with open('count.txt', 'w') as file:
        file.write(str(count))

# Function to extract face from an image
def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    for (x, y, w, h) in faces:
        cropped_face = img[y:y + h, x: x + w]
    return cropped_face

# Start video capture
cap = cv2.VideoCapture(0)
count = load_count()
speak("Please look into the camera...")

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        xcount += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Ensure the directory exists
        data_path = r"C:\Users\AMALSIVAN\OneDrive\Desktop\ibmintern (2)\ibmintern\Face1"
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        file_name_path = os.path.join(data_path, str(count) + ".jpg")
        cv2.imwrite(file_name_path, face)

        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Face Cropper", face)
    else:
        print('Face not found')
        speak("Face not found...")
        pass

    if cv2.waitKey(1) == 13 or xcount == 10:
        break

save_count(count)
cap.release()
cv2.destroyAllWindows()
print("Collecting samples complete")

# Training the model
data_path = r"C:\Users\AMALSIVAN\OneDrive\Desktop\ibmintern (2)\ibmintern\Face1"
onlyfiles = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

Training_data, Labels = [], []

for i, file in enumerate(onlyfiles):
    image_path = os.path.join(data_path, file)
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if images is not None:
        Training_data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)
    else:
        print(f"Warning: Could not read image file {image_path}")

Labels = np.asarray(Labels, dtype=np.int32)

if len(Training_data) == 0:
    print("No training data found. Exiting...")
    exit(1)

# Initialize the face recognizer and train the model
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_data), np.asarray(Labels))

# Save the trained model to a YAML file
model.save("trained_model.yml")
print("Model training complete and saved as trained_model.yml")
