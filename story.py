class Story():
    def __init__(self):
        #self.done = False
        self.current_state = "sword"
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
            },
            "fight_alongside":{
            },
            "leave":{
            },
            "toss":{
            },
            "dodge":{
            }




        }
        self.current_state = transitions.get(self.current_state, {}).get(choice, self.current_state)
        return self.get_current_scene()
    
    def introduction(self):
        return {
            "passage": "You're setting off on your own Pirate Adventure where exploration and treasures await! Somewhere beyond the horizon, "
                    "a legendary treasure lies hidden, waiting to be claimed. But treasure hunting isn't just about sailing blindly into the unknown, "
                    "you need information and strategy to outsmart the dangers that lurk ahead. With time running short, you must decide. "
                    "Do you want to talk to townspeople for information or spy on rival pirates?",
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
            "passage": 
            "You grab a cutlass from your belt and shout, Man the cannons! We fight! "
            "The crew scrambles into position, loading the cannons as the monster slams a tentacle onto the "
            "deck. The cannons roar, sending fiery iron into the monster's hide. It howls in pain, thrashing wildly. The battle rages as "
            "you hack at the tentacles with your blade, dodging as they slam into the ship. After a fierce fight, the creature lets out a "
            "final, pained wail before sinking into the sea. The crew cheers, though the ship has taken damage. You patch up the hull and "
            "continue toward Treasure Island. After days at sea the sight of land on the horizon fills your weary crew with renewed energy. "
            "Treasure Island! Your ship anchors just off the rocky shore of Treasure Island. The air is thick with the scent of sulfur, and "
            "dark smoke rises from the towering volcano at the island's center. The moment your boots hit the sand, you hear rustling from "
            "the dense jungle ahead. Do you:",
            "prompts": [
                "Investigate the rustling",
                "Head straight for the volcano caves"
            ],
            "gestures": ["left", "right"]
        }
    
    def trick(self):
        return{
            "passage": """
            You grab a barrel of gunpowder and roll it toward the monster. “Hey, ugly!” you shout, waving your arms. The creature turns, its eyes narrowing.
            You light the fuse and toss the barrel into its open maw. The explosion rocks the ship, sending the monster reeling.
            It thrashes wildly, but the crew is already working to patch the ship. You sail away, leaving the beast behind.
            After days at sea the sight of land on the horizon fills your weary crew with renewed energy. Treasure Island!
            The air is thick with the scent of sulfur, and dark smoke rises from the towering volcano at the island's center.
            The moment your boots hit the sand, you hear rustling from the dense jungle ahead. Do you:
            """,

            "prompts": [
                "Investigate the rustling",
                "Head straight for the volcano caves"
            ],
            "gestures": ["left", "right"]

        }
    
    def navigate(self):
        return{
            "passage": """
            You grab a map and chart a course through the storm. The crew works quickly, adjusting sails and securing cargo.
            The ship rocks violently as waves crash against the hull, but you keep your eyes on the horizon.
            After what feels like an eternity, the storm begins to clear. The sun breaks through the clouds, and the sea calms.
            You breathe a sigh of relief, but the crew is exhausted. They need rest before you reach Treasure Island.
            After days at sea the sight of land on the horizon fills your weary crew with renewed energy. Treasure Island!
            The air is thick with the scent of sulfur, and dark smoke rises from the towering volcano at the island's center.
            The moment your boots hit the sand, you hear rustling from the dense jungle ahead. Do you:
            """,
            "prompts": [
                "Investigate the rustling",
                "Head straight for the volcano caves"
            ],
            "gestures": ["left", "right"]
        }
    
    def investigate(self):
        return{
            "passage": """You cautiously approach the rustling, your hand on your sword. As you part the thick foliage, 
                        you discover a group of castaways—shipwrecked sailors, lost and desperate.
                        They plead for help, claiming to know the location of the treasure. But can you trust them?""",
            "prompts": [
                "Help the castaways",
                "Ignore them and press on"
            ],
            "gestures": ["left", "right"]

        }
    
    def help_casts(self):
        return{
            "passage": """You decide to help the castaways, offering them food and water. They tell you about a secret tunnel that leads to the treasure, 
                        but warn you of traps along the way. As you prepare to set off, one of them hands you a map, 
                        marking the location of the tunnel. But as you look closer, you realize it's a crude drawing, 
                        and the castaways seem nervous. Do you trust them?""",
            "prompts": [
                "Follow secret tunnel",
                "Ignore tunnel"
            ],
            "gestures": ["left", "right"]

        }
    
    def volcano(self):
        return{
            "passage": """
            You press on toward the volcano, but the jungle is thick and treacherous. 
            You stumble upon a hidden cave, its entrance shrouded in vines. 
            A low growl echoes from the darkness. This cave is not unguarded. 

        """,
            "prompts": [
                    "Draw your sword and prepare for a fight",
                    "Try and sneak in quietly"
            ],
            "gestures": ["left", "right"]

        }
    
    # def volcano(self):
    #     return{
    #         "passage": "volcano",
    #         "prompts": [
    #             "Draw your sword and prepare for a fight",
    #             "Try and sneak in quietly"
    #         ],
    #         "gestures": ["left", "right"]

    #     }
    
    def tunnel(self):
        return{
            "passage": """
            You enter the tunnel, the air thick with dust and the smell of damp earth.
            The walls are lined with strange markings, and the ground is uneven.
            As you venture deeper, you hear a faint clicking sound.
            Suddenly, a series of spears shoot out from the walls, narrowly missing you.
            You realize you've triggered a trap! Do you:""",
            "prompts": [
                "Try to disarm the trap",
                "Grab the chest and run"
            ],
            "gestures": ["left", "right"]

        }
    
    def disarm(self):
        return{
            "passage": """
            You quickly scan the walls, searching for the mechanism that triggered the trap.
            You spot a lever hidden among the markings and pull it, disabling the spears.
            The trap stops, and you breathe a sigh of relief.
            But as you turn to leave, you notice a glint of gold in the corner of the tunnel.
            A treasure chest! But it's surrounded by more traps. Do you:""",
            "prompts": [
                "Find the trigger and disable it",
                "Crawl under the spears"
            ],
            "gestures": ["left", "right"]

        }
    
    def chest_run(self):
        return{
            "passage": """
            You grab the chest and sprint down the tunnel, dodging spears as they shoot out from the walls.
            The sound of clicking grows louder, and you realize the traps are resetting.
            You reach the end of the tunnel, but the entrance collapses behind you, sealing you inside.
            You find yourself in a large chamber filled with treasure, but the walls are closing in.
            You have two choices: """,
            "prompts": [
                "Try to disarm the trap",
                "Grab the chest and run"
            ],
            "gestures": ["left", "right"]

        }
    
    def disable(self):
        return{
            "passage": "end1",
            "prompts": [
                "end"
            ],
            "gestures": []

        }
    
    def crawl(self):
        return{
            "passage": "end2",
            "prompts": [
                "end"
            ],
            "gestures": []

        }
    
    def dodge(self):
        return{
            "passage": "end3",
            "prompts": [
                "end"
            ],
            "gestures": []
        }
    
    def toss(self):
        return{
            "passage": "end4",
            "prompts": [
                "end"
            ],
            "gestures": []
        }
    
    def sword(self):
        return{
            "passage": """
            You draw your sword, ready to face the beast. The creature lunges at you, its massive jaws snapping inches from your face.
            You dodge to the side, slashing at its tentacles. The crew joins you, swords drawn and ready to fight.
            The battle is fierce, but the creature is relentless. It thrashes and roars, trying to shake you off.
            You manage to land a few blows, but the beast is strong. Do you:""",
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
            "passage": "end5",
            "prompts": [
                "end"
            ],
            "gestures": []

        }

    def leave(self):
            return{
                "passage": "end6",
                "prompts": [
                    "end"
                ],
                "gestures": []

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
            "passage": """The navigator sets the map down momentarily, distracted by an argument among the crew. This 
                        is your chance! 
                        You slip in and snatch the map, but as you retreat, a pirate catches sight of you. 
                        “Hey! Thief!” a voice shouts, and suddenly, the entire crew is after you. 
                        You dash to your ship as the pirates scramble to board theirs, they won't let this slide. 
                        The Crimson Fang is hot on your trail, their blood-red sails billowing in the wind. You have three 
                        choices to escape: """,
            "prompts": [
                "Outrun them by taking a dangerous shortcut.",
                "Ambush them by using hidden cannons or a clever trap.",
                "Negotiate or bluff your way out."
            ],
            "gestures": ["left", "right", "stop"]
        }
    
    def sneak_intel(self):
        return{
            "passage": """You listen carefully, committing every detail to memory. The Crimson Fang plans to sail through 
                        the Devil's Maw, a dangerous but fast route. 
                        As you turn to leave, you accidentally kick a loose stone—a pirate's head snaps toward you. 
                        You run for your ship, vaulting over barrels and dodging crates, but the pirates aren't far behind. 
                        By the time you raise the anchor, the Crimson Fang is already giving chase. 
                        The Crimson Fang is hot on your trail, their blood-red sails billowing in the wind. You have 
                        three choices to escape: """,
            "prompts": [
                "Outrun them by taking a dangerous shortcut.",
                "Ambush them by using hidden cannons or a clever trap.",
                "Negotiate or bluff your way out."
            ],
            "gestures": ["left", "right", "stop"]
        }
    
    def disguise(self):
            return{
                "passage": """You grab a tattered cloak and an old hat from a crate nearby, then swagger toward the pirates 
                            like you belong there. 
                            “Oi, what's all this talk about treasure?” you say, slurring your words like a drunk. 
                            The pirates laugh, assuming you're just another washed-up sailor, and unknowingly spill more 
                            details. But just as you turn to leave, one pirate eyes you suspiciously.  
                            “Wait a minute… I know you!” 
                            Before they can react, you bolt for your ship, knocking over crates to slow them down. 
                            They will not accept this, so you run to your ship while the pirates rush to board theirs. 
                            The Crimson Fang is hot on your trail, their blood-red sails billowing in the wind. You have 
                            three choices to escape:
                            """,
                "prompts": [
                    "Outrun them by taking a dangerous shortcut.",
                    "Ambush them by using hidden cannons or a clever trap.",
                    "Negotiate or bluff your way out."
                ],
                "gestures": ["left", "right", "stop"]
            }
    
    def outrun(self):
        return{
            "passage": """
            You steer your ship toward the treacherous cliffs, hoping to lose them in the narrow passages. 
            The Crimson Fang follows, but their larger ship struggles in the narrow waters. 
            A sudden gust of wind catches your sails, propelling you forward just as a massive wave slams 
            into the pirates' ship. They veer off course, barely avoiding disaster. You burst out of the Maw, 
            leaving the enemy far behind. You escape unscathed and continue toward Treasure Island. After days 
            at sea the sight of land on the horizon fills your weary crew with renewed energy. Treasure Island! 
            The air is thick with the scent of sulfur, and dark smoke rises from the towering volcano at the 
            island's center. The moment your boots hit the sand, you hear rustling from the dense jungle ahead. Do you:
            """,
            "prompts": [
                "Investigate the rustling",
                "Head straight for the volcano caves"
            ],
            "gestures": ["left", "right"]
        }
    
    def ambush(self):
        return{
            "passage": """
            You set a trap using barrels of gunpowder and hidden cannons. As the Crimson Fang sails into the narrow
            passage, you ignite the barrels, sending a fiery explosion toward them. The blast rocks their ship,
            causing chaos among the crew. You take advantage of the confusion and sail past them, escaping into the open sea.
            After days at sea the sight of land on the horizon fills your weary crew with renewed energy. Treasure Island! 
            The air is thick with the scent of sulfur, and dark smoke rises from the towering volcano at the 
            island's center. The moment your boots hit the sand, you hear rustling from the dense jungle ahead. Do you:""",
            "prompts": [
                "Investigate the rustling",
                "Head straight for the volcano caves"
            ],
            "gestures": ["left", "right"]
        }
    
    def bluff(self):
        return{
            "passage": """You raise a white flag and shout, “We surrender! We have no treasure!”
                        The Crimson Fang hesitates, unsure if you're bluffing. You take advantage of their confusion and 
                        sail away, leaving them behind. After days at sea the sight of land on the horizon fills your weary crew with renewed energy. 
                        Treasure Island! The air is thick with the scent of sulfur, and dark smoke rises from the towering volcano at the island's center. 
                        The moment your boots hit the sand, you hear rustling from the dense jungle ahead. Do you:""",
            "prompts": [
                "Investigate the rustling",
                "Head straight for the volcano caves"
            ],
            "gestures": ["left", "right"]
        }
