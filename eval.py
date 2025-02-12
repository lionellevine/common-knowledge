# eval.py
import os
import logging
import pandas as pd
from environment import Game
from agent import Player

logging.basicConfig(level=logging.INFO)

def compute_individual_banish_rate(row):
    dp = row.get("discussion_participation", 0)
    bd = row.get("banished_in_discussion", 0)
    return bd / dp if dp > 0 else None

def run_batch(num_games):
    all_game_results = []
    game_outputs = []
    
    for game_idx in range(1, num_games + 1):
        game = Game(discussion=True)
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
        results = game.play()  # List of evaluation dictionaries for this game
        all_game_results.extend(results)
        
        # Build a DataFrame from the game results and recompute banish_rate.
        df = pd.DataFrame(results)
        df["banish_rate"] = df.apply(compute_individual_banish_rate, axis=1)
        selected_keys = [
            "agent", "killer", "preprompt", "num_turns", "banished", 
            "escaped", "killed", "vote_rate_for_killer", "vote_rate_for_self", 
            "discussion_participation", "banish_rate"
        ]
        # Set the index by player name and sort alphabetically before transposing.
        df_subset = df.set_index("name")[selected_keys].sort_index().transpose()
        
        game_output = []
        header = "=" * 60 + f" Game #{game_idx} " + "=" * 60
        game_output.append(header)
        game_output.append("Player Metrics (players as columns):")
        game_output.append(df_subset.to_string())
        game_outputs.append("\n".join(game_output))
    
    return all_game_results, "\n\n".join(game_outputs)

def compute_overall_summary(all_results):
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
    num_games = 5
    all_game_results, games_output_text = run_batch(num_games)
    overall_summary_df = compute_overall_summary(all_game_results)
    
    # Use the same ordered list of prompt keys as in demo.py
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
