"""
demo.py
--------
This script demonstrates a single run of the Hoodwinked game.
It initializes a game instance, creates players, runs the game, and then
collects and outputs evaluation metrics. The script also summarizes the
prompts used during the game.
"""

import logging
from environment import Game
from agent import Player
import pandas as pd

# Set logging level to INFO for relevant output messages.
logging.basicConfig(level=logging.INFO)

def compute_individual_banish_rate(row):
    """
    Computes the banish rate for a player as the ratio of banished instances
    during discussions to the number of discussion participations.
    
    Args:
        row (pandas.Series): A row from the evaluation DataFrame for a player.
    
    Returns:
        float or None: The computed banish rate, or None if the player did not
                       participate in any discussions.
    """
    dp = row.get("discussion_participation", 0)
    bd = row.get("banished_in_discussion", 0)
    if dp > 0:
        return bd / dp
    else:
        return None

def main():
    """
    Main function that sets up and runs a single instance of the game.
    
    Steps:
      1. Creates a Game instance with discussion enabled.
      2. Instantiates several Player objects with assigned roles and prompts.
      3. Loads players into the game and runs the game loop.
      4. Collects evaluation metrics from the game run.
      5. Uses pandas to build DataFrames summarizing individual player metrics
         and overall game statistics.
      6. Also retrieves the prompts used during the game.
      7. Writes the final summary to a CSV file and prints it to the terminal.
    """
    # Create a game instance with discussion enabled.
    game = Game(discussion=True)

    # Create player instances.
    players = [
        #Player("Liam",   killer=False, preprompt="prompt_2", agent="gpt-4o-mini-2024-07-18"),
        #Player("Oliver", killer=False, preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
        Player("James",  killer=False, preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
        Player("Ezra",   killer=False, preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
        Player("Asher",  killer=False, preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
        Player("Mason",  killer=True,  preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
    ]

    # Load players into the game and start the game loop.
    game.load_players(players)
    results = game.play()  # The play() method returns a list of evaluation dictionaries.

    # Build a pandas DataFrame from the evaluation results.
    df = pd.DataFrame(results)
    df["banish_rate"] = df.apply(compute_individual_banish_rate, axis=1)

    # Section 1: Create and format a table of individual player metrics.
    selected_keys = [
        "agent", "killer", "preprompt", "num_turns", "banished",
        "killed", "vote_rate_for_killer", "vote_rate_for_self",
        "discussion_participation", "banish_rate"
    ]
    df_subset = df.set_index("name")[selected_keys].transpose()

    # Reorder columns if the desired order matches existing column names.
    desired_order = ["Mira", "Dave", "Tom", "Archie"]
    existing_names = [name for name in desired_order if name in df_subset.columns]
    if existing_names:
        df_subset = df_subset[existing_names]

    # Section 2: Compute overall summary statistics grouped by the preprompt type.
    summary_records = []
    for pre in ["prompt_1", "prompt_2"]:
        subset = df[df["preprompt"] == pre]
        count = len(subset)
        banished_count = subset["banished"].sum()
        avg_vote_rate = (subset["vote_rate_for_killer"].mean()
                         if "vote_rate_for_killer" in subset.columns and not subset["vote_rate_for_killer"].isnull().all()
                         else None)
        avg_self_vote_rate = (subset["vote_rate_for_self"].mean()
                              if "vote_rate_for_self" in subset.columns and not subset["vote_rate_for_self"].isnull().all()
                              else None)
        if "discussion_participation" in subset.columns and not subset["discussion_participation"].isnull().all():
            valid_dp = subset[subset["discussion_participation"] > 0]["discussion_participation"]
            avg_discussion = valid_dp.mean() if not valid_dp.empty else 0
        else:
            avg_discussion = 0
        valid_rates = subset["banish_rate"].dropna()
        overall_banish_rate = valid_rates.mean() if not valid_rates.empty else None
        
        summary_records.append({
            "Preprompt": pre,
            "Count": count,
            "Banished Count": banished_count,
            "Avg Vote Rate for Killer": avg_vote_rate,
            "Avg Vote Rate for Self": avg_self_vote_rate,
            "Avg Discussion Participation": avg_discussion,
            "Avg Banish Rate": overall_banish_rate
        })
    summary_df = pd.DataFrame(summary_records)

    # Section 3: Retrieve and list the prompt templates used during the game.
    ordered_keys = [
        "global_rules",
        "prompt_1",
        "identity_innocent_prompt_1",
        "identity_killer_prompt_1",
        "prompt_2",
        "identity_innocent_prompt_2",
        "identity_killer_prompt_2"
    ]
    output_lines = []
    output_lines.append("Player Metrics (players as columns):")
    output_lines.append(df_subset.to_string())
    output_lines.append("")
    output_lines.append("Overall Summary:")
    output_lines.append(summary_df.to_string(index=False))
    output_lines.append("")
    output_lines.append("Prompts Used:")
    output_lines.append("")
    for key in ordered_keys:
        prompt_text = game.prompts.get(key, f"No {key} found.")
        output_lines.append(f"{key}:")
        output_lines.append(prompt_text)
        output_lines.append("")
        
    output_text = "\n".join(output_lines)
    output_path = "results/demo.csv"
    with open(output_path, "w") as f:
        f.write(output_text)

    print(output_text)
    print(f"\nFinal evaluation metrics written to {output_path}\n")

if __name__ == "__main__":
    main()
