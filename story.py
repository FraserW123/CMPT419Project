class Story():
    def __init__(self):
        #self.done = False
        self.current_state = "introduction"
        self.states = {
            "introduction": self.introduction,
            "townspeople": self.townspeople,
            "spy": self.spy,
            "intercept_map":self.intercept_map,
            "sneak_intel":self.sneak_intel,
            "disguise":self.disguise,
            "outrun":self.outrun,
            "ambush":self.ambush,
            "bluff":self.bluff,
            "mirage": self.mirage,
            "map_path": self.map_path,
            "dream": self.dream,
            "fight_monster": self.fight_monster,
            "trick": self.trick,
            "navigate": self.navigate,
            "investigate": self.investigate,
            "volcano": self.volcano,
            "help_casts": self.help_casts,
            "tunnel":self.tunnel,
            "disarm":self.disarm,
            "chest_run":self.chest_run,
            "disable":self.disable,
            "crawl":self.crawl,
            "dodge":self.dodge,
            "toss":self.toss,
            "sword":self.sword,
            "sneak":self.sneak,
            "fight_alongside":self.fight_alongside,
            "leave":self.leave
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
                "right": "map_path",
                "stop": "dream"
            },
            "spy": {
                "left": "intercept_map",
                "right": "sneak",
                "both": "disguise"
            },
            "intercept_map": {
                "left": "outrun",
                "right": "ambush",
                "both": "bluff"
            },
            "sneak_intel": {
                "left": "outrun",
                "right": "ambush",
                "both": "bluff"
            },
            "disguise": {
                "left": "outrun",
                "right": "ambush",
                "both": "bluff"
            },
            "outrun":{
                "left": "investigate",
                "right": "volcano"
            },
            "ambush":{
                "left": "investigate",
                "right": "volcano"
            },
            "bluff":{
                "left": "investigate",
                "right": "volcano"
            },
            "mirage": {
                "left": "mirage",
                "right": "map_path",
                "stop": "dream"
            },
            "dream": {
                "left": "mirage",
                "right": "map_path",
                "stop": "dream"
            },
            "map_path":{
                "left": "fight_monster",
                "right": "trick",
                "stop": "navigate"
            },
            "fight_monster":{
                "left": "investigate",
                "right": "volcano"
            },
            "trick":{
                "left": "investigate",
                "right": "volcano"
            },
            "navigate":{
                "left": "investigate",
                "right": "volcano"
            },
            "investigate":{
                "left": "help_casts",
                "right": "volcano"
            },
            "help_casts":{
                "left": "tunnel",
                "right": "volcano"
            },
            "tunnel":{
                "left": "disarm",
                "right": "chest_run"
            },
            "disarm":{
                "left": "disable",
                "right": "crawl"
            },
            "disable":{
                "left": "disable",
                "right": "crawl"
            },
            "crawl":{
                "left": "disable",
                "right": "crawl"
            },
            "chest_run":{
                "left": "dodge",
                "right": "toss"  
            },
            "volcano":{
                "left": "sword",
                "right": "sneak"
            },
            "sword":{
                "left": "fight_alongside",
                "right": "leave"
            },
            "sneak":{
                "left": "fight_alongside",
                "right": "leave"
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
    
    def mirage(self):
        return {
            "passage": "Not quite, matey. That answer be as lost as a ship with no compass. "
            "Here is the riddle again I have seas without water, coasts without sand, towns "
            "without people and mountains without land without land. What am I?",

            "prompts": [
                "Raise your left arm for Mirage",
                "Raise your right arm for Map",
                "Raise both arms for Dream"
            ],
            "gestures": ["left", "right", "stop"]
        }
    
    def dream(self):
        return {
            "passage": "Not quite, matey. That answer be as lost as a ship with no compass. "
            "Here is the riddle again I have seas without water, coasts without sand, towns "
            "without people and mountains without land without land. What am I?",

            "prompts": [
                "Raise your left arm for Mirage",
                "Raise your right arm for Map",
                "Raise both arms for Dream"
            ],
            "gestures": ["left", "right", "stop"]
        }
    
    def map_path(self):
        return {
            "passage": "Aye, ye've got a head on yer shoulders. This here be the first clue to yer journey—Treasure Island. "
            "But beware, for the sea be filled with more than just waves. Good luck, Captain. The sea is calm as The Black Gull cuts through the waters, the wind filling your sails. The crew "
            "hums sea shanties as seagulls cry overhead. For a while, it seems like smooth sailing but then, the sky darkens. Thick storm clouds roll in, swallowing the sun. "
            "The wind shifts, and the waves begin to churn. A deep, guttural sound rises from the depths, a noise no human should ever hear. Then, with a crash, "
            "something emerges from the water. A massive creature—scales like armor, eyes like burning lanterns—rises from the sea, its long tentacles reaching for your ship. "
            "The crew stumbles back in terror as the monster roars, spraying saltwater across the deck.",

            "prompts": [
                "Raise your left arm to fight the monster",
                "Raise your right arm to trick the monster",
                "Raise both arms to navigate through the treacherous waters"
            ],
            "gestures": ["left", "right", "stop"]
        }
    
    def fight_monster(self):
        return {
            "passage": "Fighting monster",
            # "You grab a cutlass from your belt and shout, Man the cannons! We fight! "
            # "The crew scrambles into position, loading the cannons as the monster slams a tentacle onto the "
            # "deck. The cannons roar, sending fiery iron into the monster's hide. It howls in pain, thrashing wildly. The battle rages as "
            # "you hack at the tentacles with your blade, dodging as they slam into the ship. After a fierce fight, the creature lets out a "
            # "final, pained wail before sinking into the sea. The crew cheers, though the ship has taken damage. You patch up the hull and "
            # "continue toward Treasure Island. After days at sea the sight of land on the horizon fills your weary crew with renewed energy. "
            # "Treasure Island! Your ship anchors just off the rocky shore of Treasure Island. The air is thick with the scent of sulfur, and "
            # "dark smoke rises from the towering volcano at the island’s center. The moment your boots hit the sand, you hear rustling from "
            # "the dense jungle ahead. Do you:"
            "prompts": [
                "Investigate the rustling",
                "Head straight for the volcano caves"
            ],
            "gestures": ["left", "right"]
        }
    
    def trick(self):
        return{
            "passage": "trick",

            "prompts": [
                "Investigate the rustling",
                "Head straight for the volcano caves"
            ],
            "gestures": ["left", "right"]

        }
    
    def navigate(self):
        return{
            "passage": "navigate",
            "prompts": [
                "Investigate the rustling",
                "Head straight for the volcano caves"
            ],
            "gestures": ["left", "right"]
        }
    
    def investigate(self):
        return{
            "passage": "investigate",
            "prompts": [
                "Help the castaways",
                "Ignore them and press on"
            ],
            "gestures": ["left", "right"]

        }
    
    def help_casts(self):
        return{
            "passage": "help casts",
            "prompts": [
                "Follow secret tunnel",
                "Ignore tunnel"
            ],
            "gestures": ["left", "right"]

        }
    
    def ignore_casts(self):
        return{
            "passage": "help casts",
            "prompts": [
                "Follow secret tunnel",
                "Ignore tunnel"
            ],
            "gestures": ["left", "right"]

        }
    
    def volcano(self):
        return{
            "passage": "volcano",
            "prompts": [
                "Draw your sword and prepare for a fight",
                "Try and sneak in quietly"
            ],
            "gestures": ["left", "right"]

        }
    
    def tunnel(self):
        return{
            "passage": "tunnel",
            "prompts": [
                "Try to disarm the trap",
                "Grab the chest and run"
            ],
            "gestures": ["left", "right"]

        }
    
    def disarm(self):
        return{
            "passage": "disarm",
            "prompts": [
                "Find the trigger and disable it",
                "Crawl under the spears"
            ],
            "gestures": ["left", "right"]

        }
    
    def chest_run(self):
        return{
            "passage": "tunnel",
            "prompts": [
                "Try to disarm the trap",
                "Grab the chest and run"
            ],
            "gestures": ["left", "right"]

        }
    
    def disable(self):
        return{
            "passage": "end",
            "prompts": [
                "Find the trigger and disable it",
                "Crawl under the spears"
            ],
            "gestures": ["left", "right"]

        }
    
    def crawl(self):
        return{
            "passage": "end",
            "prompts": [
                "Find the trigger and disable it",
                "Crawl under the spears"
            ],
            "gestures": ["left", "right"]

        }
    
    def dodge(self):
        return{
            "passage": "end",
            "prompts": [
                "Find the trigger and disable it",
                "Crawl under the spears"
            ],
            "gestures": ["left", "right"]

        }
    
    def toss(self):
        return{
            "passage": "end",
            "prompts": [
                "Find the trigger and disable it",
                "Crawl under the spears"
            ],
            "gestures": ["left", "right"]

        }
    
    def sword(self):
        return{
            "passage": "sword",
            "prompts": [
                "Fight alongside your crew",
                "Leave your crew to distract the beast"
            ],
            "gestures": ["left", "right"]

        }
    
    def sneak(self):
        return{
            "passage": "sneaking",
            "prompts": [
                "Fight alongside your crew",
                "Leave your crew to distract the beast"
            ],
            "gestures": ["left", "right"]

        }

    def fight_alongside(self):
        return{
            "passage": "end",
            "prompts": [
                "Fight alongside your crew",
                "Leave your crew to distract the beast"
            ],
            "gestures": ["left", "right"]

        }

    def leave(self):
            return{
                "passage": "end",
                "prompts": [
                    "Fight alongside your crew",
                    "Leave your crew to distract the beast"
                ],
                "gestures": ["left", "right"]

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
    
    def intercept_map(self):
        return{
            "passage": "Intercepting map",
            "prompts": [
                "Outrun them",
                "Ambush them",
                "Negotiate or bluff them"
            ],
            "gestures": ["left", "right", "stop"]
        }
    
    def sneak_intel(self):
        return{
            "passage": "sneaking for intel",
            "prompts": [
                "Outrun them",
                "Ambush them",
                "Negotiate or bluff them"
            ],
            "gestures": ["left", "right", "stop"]
        }
    
    def disguise(self):
            return{
                "passage": "using a disguise",
                "prompts": [
                    "Outrun them",
                    "Ambush them",
                    "Negotiate or bluff them"
                ],
                "gestures": ["left", "right", "stop"]
            }
    
    def outrun(self):
        return{
            "passage": "outrunning",
            "prompts": [
                "Investigate the rustling",
                "Head straight for the volcano caves"
            ],
            "gestures": ["left", "right"]
        }
    
    def ambush(self):
        return{
            "passage": "ambush",
            "prompts": [
                "Investigate the rustling",
                "Head straight for the volcano caves"
            ],
            "gestures": ["left", "right"]
        }
    
    def bluff(self):
        return{
            "passage": "bluffing",
            "prompts": [
                "Investigate the rustling",
                "Head straight for the volcano caves"
            ],
            "gestures": ["left", "right"]
        }
