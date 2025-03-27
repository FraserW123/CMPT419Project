# Import necessary modules
import qi  # For interacting with NAOqi
import sys  # To interact with system-specific parameters

def main():
    # Connect to the Pepper robot's NAOqi framework
    try:
        # Replace with your Pepper's IP address
        pepper_ip = "10.0.0.14"
        port = 9559

        # Connect to the NAOqi framework
        app = qi.Application(["--ip", pepper_ip, "--port", str(port)])
        app.start()

        # Create an ALTextToSpeech proxy to use text-to-speech capabilities
        tts = app.session.service("ALTextToSpeech")

        # Create an ALMotion proxy to control Pepper's movement
        motion = app.session.service("ALMotion")

        # Make Pepper say a phrase
        tts.say("Hello, I am Pepper. Watch me wave!")

        # Move Pepper's right arm to simulate a wave
        wave_arm(motion)

    except RuntimeError:
        print("Failed to connect to the robot at IP address:", pepper_ip)
        sys.exit(1)

def wave_arm(motion):
    # Set the right arm's positions to simulate a waving gesture
    # First, move to an initial position (e.g., arm down)
    motion.setAngles(["RShoulderPitch", "RElbowYaw", "RElbowRoll"], [-1.5, 0.0, 0.0], 0.2)
    motion.setAngles(["RShoulderRoll", "RWristYaw"], [0.0, 0.0], 0.2)
    
    # Pause briefly
    time.sleep(1)

    # Move arm up for waving
    motion.setAngles(["RShoulderPitch", "RElbowYaw", "RElbowRoll"], [-0.5, 0.5, 0.0], 0.2)
    motion.setAngles(["RShoulderRoll", "RWristYaw"], [0.3, 0.5], 0.2)
    
    # Pause briefly
    time.sleep(1)

    # Move arm back to neutral position (resting position)
    motion.setAngles(["RShoulderPitch", "RElbowYaw", "RElbowRoll"], [-1.5, 0.0, 0.0], 0.2)
    motion.setAngles(["RShoulderRoll", "RWristYaw"], [0.0, 0.0], 0.2)

if __name__ == "__main__":
    main()
