# Human-Robot Interactions through Interactive Stories
## Fraser Wong, Lazar Pajic, Nathan Chow



## Overview

This project explores how arm gestures performed by the Pepper robot can influence user engagement and immersion in interactive storytelling. We developed a gesture-enhanced storytelling application for Pepper, integrating arm movement cues during key narrative moments. A user study was conducted to evaluate the effects of these gestures, comparing scenes with and without physical gestures through participant feedback.

The study involved 20 participants and used both qualitative and quantitative survey responses to assess engagement, interactivity, and immersion. The results indicated that while gestures significantly improved user engagement and attention, their impact on narrative immersion was more varied.

![](images/all3.png)

## Features

- Gesture-enhanced storytelling using Pepper's arms
- Custom interaction design with and without gestures
- Real-time gesture recognition and logging
- User study data collection and analysis
- Demo video and survey feedback integration

## Technologies Used

- Pepper robot with qi SDK
- Python (gesture logic and control scripts)
- CSV-based logging and dataset management
- Survey data collection (Google Forms)
- Video recording and editing tools for demo
- Linux-based operating system was used to run code that connects to pepper

## Folder Structure
```bash
project-root/
├── analysis/                       # Stores hypothesis testing files and results
├── consent forms/                  # Consent forms of participants
├── images/                         # Stores images
├── model training/                 # Stores training files
├── arm_gesture_landmark_model.pth  # Gesture classification model
├── pepper.py                       # Code to connect to pepper
├── story.py                        # Narrative content file
├── requirements.txt                # Required libraries
└── README.md
```

## Setup Instructions

1. **Install Dependencies**
    Make sure the development is set up with Python 3 as well as the dependencies listed in the provided requirements.txt file.

2. **Connect to Pepper**
    Connect your computer to the same network as Pepper and retreive its IP address. Enter in the necessary credentials to run code.

3. **Run the Application**
    Use the provided control script `pepper.py` to launch the storytelling session.
    ```bash
    python3 pepper.py
    ```
4.  **Participant Study**
    Follow the scripted interaction for each participant and collect gesture recognition data which is automatically saved to CSV.

5. **Survey**
    Administer the follow-up survey either digitally or in print to gather user feedback.

## Training Setup
Our main training script `model training/gesture_landmark_classification.ipynb` contains the code used to build and train gesture classification models based on landmark data. During development, we also explored an alternative model which identifies gestures directly from images. However, we ultimately chose to focus on the landmark-based approach due to its higher accuracy and more reliable performance. The image-based model remains a promising direction for future work, particularly with further optimization and a larger dataset.


## Acknowledgements

To connect to the Pepper robot using the qi library, we referenced this stack overflow post:

https://stackoverflow.com/questions/77987028/how-can-i-connect-to-pepper-naoqi-2-9-via-libqi-python 

To train our model, we used this dataset from kaggle:

https://www.kaggle.com/datasets/williamsvealochert/arm-and-body-gestureswrss 

We would also like to thank our TA Shay Zhang for onboarding us on how to use the pepper robots.


## Self Evaluation

Our initial proposal was motivated to create more emotionally intelligent and engaging interactions between humans and robots. We planned to do this by optimizing the story based on player engagement, as well as recommending story paths to players during the game. However, we realized these plans were overly ambitious given our timeline, so we focused on having the robot perform arm gestures and enabling players to use their own gestures to make choices during the game. This shift allowed us to build a functional and focused prototype that still captured our core goal of enhancing human-robot interactions. 