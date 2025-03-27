import time
import qi
import sys
import numpy as np

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

# Connect to the robot 
app = qi.Application(sys.argv, url="tcps://10.0.0.14:9503")
logins = ("nao", "nao")
factory = AuthenticatorFactory(*logins)
app.session.setClientAuthenticatorFactory(factory)
app.start()
print("started")

tts = app.session.service("ALTextToSpeech")
motion = app.session.service("ALMotion")
tts.say("Arm movements")

poses = ["left", "right", "T pose"]


def wave_arm(motion, pose):
    # Set the right arm's positions to simulate a waving gesture
    # First, arm to the right
    # motion.setAngles(["RShoulderPitch", "RElbowYaw", "RElbowRoll"], [-0.4, 0.0, 0.0], 0.2)
    # motion.setAngles(["RShoulderRoll", "RWristYaw"], [-1, 0.0], 0.2)
    
    # Pause briefly
    # time.sleep(7)
    tts.say(pose)
    if pose == "T pose":
      motion.setAngles(["RShoulderPitch", "LShoulderPitch", "RShoulderRoll", "LShoulderRoll"], [-0.75, -0.75,-1,1], 0.4)
    elif pose == "left":
      motion.setAngles(["LShoulderPitch", "LShoulderRoll"], [-0.75,1], 0.4)
    elif pose == "right":
      motion.setAngles(["RShoulderPitch", "RShoulderRoll"], [-0.75,-1], 0.4)
        



def move_body(motion):
    motion.moveTo(-0.5,0.0,0.0)

for pose in poses:
  wave_arm(motion, pose)
  time.sleep(8)

tts.say("All done!")

# motion.setStiffnesses("Head", 1.0)
# motion.setAngles("HeadYaw", 0.5, 0.05)
# motion.setAngles("HeadPitch", 0.5, 0.05)


