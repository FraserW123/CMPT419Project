class Story():
    def __init__(self):
        #self.done = False
        self.current_state = "introduction"
        self.states = {
            "introduction": self.introduction,
            "townspeople": self.townspeople,
            "spy": self.spy,
            # "mirage": self.mirage,
            # "map": self.map_path,
            # "dream": self.dream
        }

    def get_current_scene(self):
        return self.states[self.current_state]()
    
    def transition(self, choice):
        transitions = {
            "introduction": {
                "left": "townspeople",
                "right": "spy"
            },
            "townspeople": {
                "left": "mirage",
                "right": "map",
                "stop": "dream"
            },
            "spy": {
                "left": "intercept_map",
                "right": "sneak",
                "both": "disguise"
            }
        }
        self.current_state = transitions.get(self.current_state, {}).get(choice, self.current_state)
        return self.get_current_scene()
    
    def introduction(self):
        return {
            "passage": "Intro",#"You're setting off on your own Pirate Adventure where exploration and treasures await! Somewhere beyond the horizon, a legendary treasure lies hidden, waiting to be claimed. But treasure hunting isn't just about sailing blindly into the unknown, you need information and strategy to outsmart the dangers that lurk ahead. " \
                    #"With time running short, you must decide. Do you want to talk to townspeople for information or spy on rival pirates?",
            "prompts": [
                "Raise your left arm to talk to townspeople",
                "Raise your right arm to spy on rivals"
            ],
            "gestures": ["left", "right"]
        }
    
    def townspeople(self):
        return {
            "passage": "The locals here know more than they let on.Sailors, merchants, and old sea dogs whisper tales " \
                    "of fortune, danger, and secrets buried in riddles. " \
                    "Perhaps one of them can point you in the right direction. You step into a dimly lit tavern, where an old sailor " \
                    "with a glass eye beckons you closer. He offers a riddle, hinting at a safe but unpredictable voyage. Solve it, " \
                    "and you'll have a solid course to follow. He clears his throat and begins. " \
                    "I have seas without water, " \
                    "coasts without sand, " \
                    "towns without people, and " \
                    "mountains without land. " \
                    "What am I?",
            "prompts": [
                "Raise your left arm for Mirage",
                "Raise your right arm for Map",
                "Raise both arms for Dream"
            ],
            "gestures": ["left", "right", "stop"]
        }
    
    def spy(self):
        return{
            "passage": "The Crimson Fang, a ruthless crew of cutthroats, has been searching for the treasure as well. If" \
                "anyone knows where to find it, it's them but they don't share secrets willingly. You creep" \
                "through the shadows of the docks, eavesdropping on their captain's conversation. You notice"\
                "their navigator holding the map to Treasure Island!",
            "prompts": [
                "Intercept the map",
                "Sneak away with the intel",
                "Use a disguise"
            ],
            "gestures": ["left", "right", "stop"]
        }
