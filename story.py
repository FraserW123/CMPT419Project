class Story():
    def __init__(self):
        self.done = False
    
    def introduction(self):
        passage = "You're setting off on your own Pirate Adventure where exploration and treasures await! Somewhere beyond the horizon, a legendary treasure lies hidden, waiting to be claimed. But treasure hunting isn't just about sailing blindly into the unknown, you need information and strategy to outsmart the dangers that lurk ahead. " \
        "With time running short, you must decide. Do you want to talk to townspeople for information or spy on rival pirates?"
        
        path1 = "Raise your left arm if you want to talk to townspeople for information?"
        path2 = "Or raise your right arm if you want to spy on rival pirates?"

        return passage, path1, path2, None
    
    def townspeople(self):
        passage = "The locals here know more than they let on. Sailors, merchants, and old sea dogs whisper tales " \
        "of fortune, danger, and secrets buried in riddles. " \
        "Perhaps one of them can point you in the right direction. You step into a dimly lit tavern, where an old sailor " \
        "with a glass eye beckons you closer. He offers a riddle, hinting at a safe but unpredictable voyage. Solve it, " \
        "and you'll have a solid course to follow. He clears his throat and begins. " \
        "I have seas without water, " \
        "coasts without sand, " \
        "towns without people, and " \
        "mountains without land. " \
        "What am I?"

        path1 = "A mirage?"
        path2 = "A map?"
        path3 = "Or a dream?"

        return passage, path1, path2, path3
    
    def spy(self):
        passage = "The Crimson Fang, a ruthless crew of cutthroats, has been searching for the treasure as well. If" \
                "anyone knows where to find it, it's them but they don't share secrets willingly. You creep" \
                "through the shadows of the docks, eavesdropping on their captain's conversation. You notice"\
                "their navigator holding the map to Treasure Island!"\
                
        return passage
