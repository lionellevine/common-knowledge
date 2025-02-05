# demo.py
import logging
from environment import Game
from agent import Player

logging.basicConfig(level=logging.INFO)

def main():
    # Create the game
    game = Game(discussion=True)

    # Create players
    players = [
        Player("Bob",    killer=False, preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
        Player("Jim",    killer=True,  preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
        Player("Dave",   killer=False, preprompt="prompt_2", agent="gpt-4o-mini-2024-07-18"),
        Player("Mira",   killer=True,  preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
        Player("Archie", killer=False, preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
        Player("Tom",    killer=True,  preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
        Player("Sarah",  killer=False, preprompt="prompt_2", agent="gpt-4o-mini-2024-07-18"),
        Player("Gabriel",killer=True,  preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
        Player("Derek",  killer=False, preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18")
    ]

    game.load_players(players)
    results = game.play()

    print("Game Over! Final evaluation metrics:")
    for r in results:
        print(r)

if __name__ == "__main__":
    main()
