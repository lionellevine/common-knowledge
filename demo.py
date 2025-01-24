from agent import Player
from environment import Game

# Define the game
game = Game()

# Load the players into the game
game.load_players([
    Player("Bob", killer=False, preprompt="rules", agent="gpt-3.5"),
#    Player("Adam", killer=True, agent="cli"),
    Player("Jim", killer=False, preprompt="rules_alt", agent="gpt-3.5"),
    Player("Dave", killer=False, preprompt="rules", agent="gpt-3.5"),
    Player("Mira", killer=True, preprompt="rules", agent="gpt-3.5"),
])

# Play the game
game.play()