import qi

# Replace with your Pepper's IP
PEPPER_IP = "10.0.0.14"
PORT = 9559

def main():
    # Connect to Pepper's session
    session = qi.Session()
    try:
        session.connect(f"tcp://{PEPPER_IP}:{PORT}")
    except RuntimeError:
        print("Can't connect to Pepper. Check the IP and port.")
        return

    # Create a text-to-speech service proxy
    tts = session.service("ALTextToSpeech")
    # motion = session.service("ALMotion")
    tts.say("Hello! I am Pepper.")  # Make Pepper say something

    # Move Pepper's head
    # motion.setStiffnesses("Head", 1.0)
    # motion.setAngles("HeadYaw", 0.5, 0.5)
    # motion.setAngles("HeadPitch", 0.5, 0.5)

    # # Move Pepper's arms
    # motion.setStiffnesses("RArm", 1.0)
    # motion.setAngles("RShoulderPitch", 1.0, 0.5)
    # motion.setAngles("RShoulderRoll", 0.5, 0.5)
    # motion.setAngles("RElbowYaw", 1.0, 0.5)
    # motion.setAngles("RElbowRoll", 0.5, 0.5)
    # motion.setAngles("RWristYaw", 1.0, 0.5)

    

if __name__ == "__main__":
    main()
