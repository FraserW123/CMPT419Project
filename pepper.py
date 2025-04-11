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
from story import Story

import torch
import torch.nn as nn

# Initialize CSV with participant id column
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


# Camera setup
resolution = 3  # VGA (640x480)
colorSpace = 13  # BGR
fps = 25
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
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

def signal_left(motion):
    motion.setStiffnesses("LArm", 1.0)
    motion.setAngles(["LShoulderPitch", "LShoulderRoll"], [-0.75,1], 0.5)

def signal_right(motion):
    motion.setStiffnesses("RArm", 1.0)
    motion.setAngles(["RShoulderPitch", "RShoulderRoll"], [-0.75,-1], 0.5)

def signal_both(motion):
    motion.setAngles(["RShoulderPitch", "LShoulderPitch", "RShoulderRoll", "LShoulderRoll"], [-0.75, -0.75,-1,1], 0.5)

# Detects Left, Right, and Both arms gestures using mediapipe
# Checks to see if wrists is above shoulders
def detect_gesture(landmarks):
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
            input_tensor = torch.tensor(landmark_vector, dtype=torch.float32).unsqueeze(0)
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            conf, predicted = torch.max(probabilities, 1)
            
            # Add confidence threshold
            if conf < 0.98:  # Adjust threshold as needed
                return None
        
        # Map model output to labels
        label_map = {0: "left", 1: "right", 2: "stop"}
        return label_map[predicted.item()]
        
    except Exception as e:
        logging.error(f"Model prediction failed: {str(e)}")
        return None

# Choose your own adventure story
story_mode = Story()
current_scene = story_mode.get_current_scene()

# Gesture Detection Cooldown 
LAST_DETECTION_TIME = 0
COOLDOWN = 3.0
no_gesture = 1

try:
    motion.setStiffnesses("Head", 0.0)  # Relax head motors
    done = False
    # Main loop for the Pepper application
    while not done:
        tts.say(current_scene["passage"])

        # Checks if current scene is an ending scene
        if current_scene["prompts"][0] == "end":
            tts.say("Thank you!")
            done = True
            break

        # Logic to control if pepper performs arm gestures during current scene
        # (Current: skips every 3rd scene)
        count = 0
        for prompt in current_scene["prompts"]:
            tts.say(prompt)
            
            if no_gesture % 3 != 0:
                if count == 0:
                    signal_right(motion)
                elif count == 1:
                    signal_left(motion)
                else:
                    signal_both(motion)
                count+=1
            
            time.sleep(5)
        
        tts.say("Make your choice now")
        time.sleep(2)

        # Gesture detection logic
        gesture = None
        start_time = time.time()
        while time.time() - start_time < 75:  # 75 second timeout 
            # Capture frame and detect gesture
            if gesture in current_scene["gestures"]:
                current_scene = story_mode.transition(gesture)
                no_gesture += 1
                break
            
            # Captures frame from Pepper's camera if available
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

            # If valid gesture was detected save it to dataset
            if results.pose_landmarks:
                # Draw landmarks - adds the landmarks on the camera pop up
                mp_drawing.draw_landmarks(
                    img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                current_time = time.time()
                if (current_time - LAST_DETECTION_TIME) > COOLDOWN:
                    # Detect gesture
                    gesture = detect_gesture(results.pose_landmarks.landmark)

                    # Get raw landmark data
                    raw_landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]
                    
                    if gesture and gesture in current_scene["gestures"]:
                        LAST_DETECTION_TIME = current_time
                        response = gesture
                        tts.say(response)
                        logging.info(f"Detected gesture: {gesture}")
                        logging.info(f"Pepper responded: {response} to gesture: {gesture}")

                        # Save to CSV with timestamp
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        csv_writer.writerow([timestamp, participant_id, gesture, str(raw_landmarks)])
                        csv_file.flush()  # Ensure immediate write

            else: 
                logging.warning("No landmarks detected in frame")

            # Add text overlay for detected gesture
            cv2.putText(img, f"Gesture: {gesture}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display preview
            cv2.imshow('Pepper Camera', img)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break

finally:
    csv_file.close()
    video_service.unsubscribe(subscriberId)
    motion.setStiffnesses("Head", 1.0)  # Relax head motors
    cv2.destroyAllWindows()
    print("Disconnected")