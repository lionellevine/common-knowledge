"""
eval.py
---------
This module runs a batch of game simulations for the Hoodwinked game and aggregates evaluation metrics.
It computes individual player metrics (such as banish rates, vote rates, and discussion participation)
and then summarizes the results in tabular form. Finally, it outputs the evaluation results along with
the prompts used during the game to a CSV file.
"""

import os
import logging
import pandas as pd
from environment import Game
from agent import Player

# Set up logging for the evaluation process.
logging.basicConfig(level=logging.INFO)

def compute_individual_banish_rate(row):
    """
    Computes the banish rate for an individual player.
    
    The banish rate is defined as the ratio of times a player was banished during discussions
    to the number of discussion participations.
    
    Args:
        row (pandas.Series): A row from the DataFrame containing a player's evaluation metrics.
    
    Returns:
        float or None: The banish rate, or None if the player did not participate in any discussions.
    """
    dp = row.get("discussion_participation", 0)
    bd = row.get("banished_in_discussion", 0)
    return bd / dp if dp > 0 else None

def run_batch(num_games):
    """
    Runs multiple game simulations and collects evaluation metrics.
    
    For each game:
      - A Game instance is created (with discussion enabled).
      - A fixed set of players is created and loaded into the game.
      - The game is played, and each player's evaluation metrics are collected.
      - The evaluation results are converted into a pandas DataFrame.
      - A subset of the metrics (e.g., vote rates, discussion participation, etc.) is extracted
        and transposed to form a player metrics table.
      - A summary output for the game is generated.
    
    Args:
        num_games (int): The number of game simulations to run.
    
    Returns:
        tuple: (all_game_results, game_outputs)
            all_game_results (list): List of evaluation dictionaries (one per player, from all games).
            game_outputs (str): A concatenated string summary of each game.
    """
    all_game_results = []
    game_outputs = []
    
    for game_idx in range(1, num_games + 1):
        # Create a game instance with discussion enabled.
        game = Game(discussion=True)
        # Create a fixed set of players.
        players = [
            Player("Liam",    killer=False, preprompt="prompt_2", agent="gpt-4o-mini-2024-07-18"),
            Player("Noah",    killer=False, preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
            Player("Oliver",  killer=False, preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
            Player("James",   killer=False, preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
            Player("Elijah",  killer=False, preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
            Player("William", killer=False, preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
            Player("Benjamin",killer=False, preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
            Player("Lucas",   killer=False, preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
            Player("Henry",   killer=False, preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
            Player("Jacob",   killer=False, preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
            Player("Matthew", killer=False, preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
            Player("Tom",     killer=True,  preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18")
        ]
        game.load_players(players)
        # Play the game and collect the evaluation metrics.
        results = game.play()  # Each game returns a list of evaluation dictionaries.
        all_game_results.extend(results)
        
        # Create a DataFrame from the results.
        df = pd.DataFrame(results)
        df["banish_rate"] = df.apply(compute_individual_banish_rate, axis=1)
        # Select relevant keys for the metrics table.
        selected_keys = [
            "agent", "killer", "preprompt", "num_turns", "banished", 
            "killed", "vote_rate_for_killer", "vote_rate_for_self", 
            "discussion_participation", "banish_rate"
        ]
        # Transpose the DataFrame so that players are shown as columns.
        df_subset = df.set_index("name")[selected_keys].sort_index().transpose()
        
        game_output = []
        header = "=" * 60 + f" Game #{game_idx} " + "=" * 60
        game_output.append(header)
        game_output.append("Player Metrics (players as columns):")
        game_output.append(df_subset.to_string())
        game_outputs.append("\n".join(game_output))
    
    # Return both the raw evaluation results and the string summary of all games.
    return all_game_results, "\n\n".join(game_outputs)

def compute_overall_summary(all_results):
    """
    Computes an overall summary of evaluation metrics across all game simulations.
    
    The summary is grouped by the preprompt type (e.g., "prompt_1" vs. "prompt_2") and includes:
      - Count of players
      - Number of times banished
      - Average vote rate for the killer
      - Average vote rate for self
      - Average discussion participation
      - Average banish rate
    
    Args:
        all_results (list): List of evaluation dictionaries from all game simulations.
    
    Returns:
        pandas.DataFrame: A DataFrame summarizing the aggregated metrics by preprompt type.
    """
    df_all = pd.DataFrame(all_results)
    df_all["banish_rate"] = df_all.apply(compute_individual_banish_rate, axis=1)
    summary_records = []
    for pre in ["prompt_1", "prompt_2"]:
        subset = df_all[df_all["preprompt"] == pre]
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
    return pd.DataFrame(summary_records)

def main():
    """
    Main function to run a batch of game simulations and output the evaluation metrics.
    
    Steps:
      1. Run a specified number of game simulations.
      2. Compute individual evaluation metrics for each player.
      3. Generate a summary table of player metrics.
      4. Compute an overall summary of the metrics grouped by preprompt type.
      5. Retrieve the game prompts used during the simulation.
      6. Write the full evaluation output (player metrics, overall summary, and prompts)
         to a CSV file and print it.
    """
    num_games = 5
    all_game_results, games_output_text = run_batch(num_games)
    overall_summary_df = compute_overall_summary(all_game_results)
    
    # Create a temporary game instance to retrieve prompt templates.
    temp_game = Game(discussion=True)
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
    output_lines.append(games_output_text)
    output_lines.append("")
    output_lines.append("Overall Summary:")
    output_lines.append(overall_summary_df.to_string(index=False))
    output_lines.append("")
    output_lines.append("Prompts Used:")
    output_lines.append("")
    # Retrieve and list each prompt template.
    for key in ordered_keys:
        prompt_text = temp_game.prompts.get(key, f"No {key} found.")
        output_lines.append(f"{key}:")
        output_lines.append(prompt_text)
        output_lines.append("")
        
    final_output_text = "\n".join(output_lines)
    output_path = os.path.join("results", "final_evaluation.csv")
    with open(output_path, "w") as f:
        f.write(final_output_text)
    
    print(final_output_text)
    print(f"\nFinal evaluation metrics written to {output_path}\n")

if __name__ == "__main__":
    main()
