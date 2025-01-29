import logging

from environment import Game
from agent import Player

logging.basicConfig(level=logging.INFO)

def main():
    # Create the game
    game = Game(discussion=True)

    # Create players
    players = [
        Player("Bob",  killer=False, preprompt="rules",      agent="gpt-4o-mini-2024-07-18"),
        Player("Jim",  killer=False, preprompt="rules_alt",  agent="gpt-4o-mini-2024-07-18"),
        Player("Dave", killer=False, preprompt="rules",      agent="gpt-4o-mini-2024-07-18"),
        Player("Mira", killer=True,  preprompt="rules",      agent="gpt-4o-mini-2024-07-18"),
        Player("Harry",  killer=False, preprompt="rules_alt",  agent="gpt-4o-mini-2024-07-18"),
        Player("Snape",  killer=True, preprompt="rules_alt",  agent="gpt-4o-mini-2024-07-18"),
        Player("Ron",  killer=False, preprompt="rules_alt",  agent="gpt-4o-mini-2024-07-18")
    ]

    # Load them into the game
    game.load_players(players)

    # Play!
    results = game.play()

    print("Game Over! Final evaluation metrics:")
    for r in results:
        print(r)

    # -------------------------------------------------------------------------
    # NEW: Compare how well "rules" vs "rules_alt" innocents guessed the killer
    # -------------------------------------------------------------------------
    group_sums = {"rules": 0.0, "rules_alt": 0.0}
    group_counts = {"rules": 0, "rules_alt": 0}

    for r in results:
        # Only examine innocents
        if not r["killer"]:
            grp = r["preprompt"]
            # Only tally groups we know about
            if grp in group_sums:
                group_sums[grp] += r.get("vote_rate_for_killer", 0.0)
                group_counts[grp] += 1

    print("\n=== Did 'rules' or 'rules_alt' guess the killer more accurately? ===")
    for grp in group_sums:
        if group_counts[grp] > 0:
            avg = group_sums[grp] / group_counts[grp]
            print(f"Group '{grp}': average vote_rate_for_killer = {avg:.2f}")
        else:
            print(f"Group '{grp}': no data (no surviving innocents or no votes)")

if __name__ == "__main__":
    main()
