import qi
import sys
import time
import cv2
import numpy as np
import mediapipe as mp
import logging
import csv
from datetime import datetime
import os

import torch
import torch.nn as nn

# Initialize CSV with participant columns (change participant column to just an ID column)
participant_id = input("Enter participant ID: ").strip() or "unknown"

# CSV 
csv_file = open('gesture_dataset.csv', 'a', newline='')
csv_writer = csv.writer(csv_file)
if os.stat('gesture_dataset.csv').st_size == 0:
    csv_writer.writerow(['timestamp', 'participant_id', 
                    'gesture', 'landmarks'])

# Logging settings
logging.basicConfig(
    filename='gesture_logs.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Authenticator:
    def __init__(self, username, password):
        self.username = username
        self.password = password

    # This method is required by libqi and should return a dictionary with login information.
    # The dictionary should have the keys 'user' and 'token'.
    def initialAuthData(self):
        return {'user': self.username, 'token': self.password}

class AuthenticatorFactory:
    def __init__(self, username, password):
        self.username = username
        self.password = password

    # This method is required by libqi and should return an object that has at least
    # the `initialAuthData` method. It is used for authentication.
    def newAuthenticator(self):
        return Authenticator(self.username, self.password)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Connect to the robot 
app = qi.Application(sys.argv, url="tcps://10.0.0.6:9503")
logins = ("nao", "nao")
factory = AuthenticatorFactory(*logins)
app.session.setClientAuthenticatorFactory(factory)
app.start()
print("Connected to Pepper")

# Get services
video_service = app.session.service("ALVideoDevice")
tts = app.session.service("ALTextToSpeech")
motion = app.session.service("ALMotion")
autonomous_life = app.session.service("ALAutonomousLife")

autonomous_life.setState("disabled")  # Disable autonomous behaviors

# Set Pepper's head to look up (HeadPitch angle in radians)
motion.setStiffnesses("Head", 1.0)  # Enable motor stiffness
motion.setAngles("HeadPitch", -0.5, 0.2)  # Tilt head up (adjust -0.3 as needed)

# Camera setup
resolution = 3  # VGA (640x480)
colorSpace = 13  # BGR
fps = 15
cameraId = 0  # Top camera
subscriberId = video_service.subscribeCamera("videoSubscriber", cameraId, resolution, colorSpace, fps)

# Arm gesture model class definition
class ArmGestureNet(nn.Module):
    def __init__(self, input_size=18, num_classes=3):
        super(ArmGestureNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.4)
        self.batchnorm = nn.BatchNorm1d(128)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Add after MediaPipe initialization
model_path = "arm_gesture_landmark_model.pth"
model = ArmGestureNet(input_size=18)
model.load_state_dict(torch.load(model_path))
model.eval()

# Detects Left, Right, and Both arms gestures using mediapipe
# Checks to see if wrists is above shoulders
def detect_gesture(landmarks):
    """ try:
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Check vertical positions (MediaPipe y decreases upward)
        left_up = left_wrist.y < left_shoulder.y - 0.1  # Added margin
        right_up = right_wrist.y < right_shoulder.y - 0.1

        logging.debug(f"Left wrist Y: {left_wrist.y}, Shoulder Y: {left_shoulder.y}")

        if left_up and right_up:
            return "both"
        elif left_up:
            return "left"
        elif right_up:
            return "right"
        return None
    except Exception as e:
        logging.error(f"Gesture detection failed: {str(e)}")
        return None """
    try:
        # Extract the same landmarks used in training
        landmark_indices = [
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST,
            mp_pose.PoseLandmark.RIGHT_WRIST
        ]
        
        landmark_vector = []
        for idx in landmark_indices:
            lmk = landmarks[idx]
            landmark_vector.extend([lmk.x, lmk.y, lmk.z])
        
        # Convert to tensor and predict
        with torch.no_grad():
            input_tensor = torch.tensor(landmark_vector, dtype=torch.float32)
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=0)
            conf, predicted = torch.max(probabilities, 0)
            
            # Add confidence threshold
            if conf < 0.8:  # Adjust threshold as needed
                return None
        
        # Map model output to labels
        label_map = {0: "left", 1: "right", 2: "stop"}
        return label_map[predicted.item()]
        
    except Exception as e:
        logging.error(f"Model prediction failed: {str(e)}")
        return None

# Choose your own adventure story
story = {
    "start": {
        "text": "Welcome to the adventure! Raise your left or right arm to choose your path.",
        "choices": {
            "left": "You chose the left path! A dragon appears...",
            "right": "You chose the right path! A treasure awaits...",
            "stop": "Secret path unlocked! You find a magic portal."
        }
    }
}

try:
    current_state = "start"
    tts.say(story[current_state]["text"])
    
    while True:
        gesture = None
        naoImage = video_service.getImageRemote(subscriberId)
        if naoImage is None:
            logging.error("Failed to capture frame from Pepper's camera")
            continue

        # Convert image to OpenCV format
        width, height = naoImage[0], naoImage[1]
        array = naoImage[6]
        img = np.frombuffer(array, dtype=np.uint8).reshape((height, width, 3))
        
        # Convert BGR to RGB for MediaPipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = pose.process(rgb_img)
        

        if results.pose_landmarks:
            # Draw landmarks (optional) - adds the landmarks on the camera pop up
            # mp_drawing.draw_landmarks(
            #     img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Detect gesture
            gesture = detect_gesture(results.pose_landmarks.landmark)

            # Get raw landmark data
            raw_landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]
            
            if gesture and gesture in story[current_state]["choices"]:
                response = story[current_state]["choices"][gesture]
                tts.say(response)
                logging.info(f"Detected gesture: {gesture}")
                logging.info(f"Pepper responded: {response} to gesture: {gesture}")

                # Save to CSV with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                csv_writer.writerow([timestamp, participant_id, gesture, str(raw_landmarks)])
                csv_file.flush()  # Ensure immediate write

                break  # Exit after first choice (or modify for multi-step story)
        else: 
            logging.warning("No landmarks detected in frame")

        # Add text overlay for detected gesture
        cv2.putText(img, f"Gesture: {gesture}", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display preview (optional)
        cv2.imshow('Pepper Camera', img)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

finally:
    csv_file.close()
    video_service.unsubscribe(subscriberId)
    motion.setStiffnesses("Head", 0.0)  # Relax head motors
    cv2.destroyAllWindows()
    print("Disconnected")