# demo.py
import logging
from environment import Game
from agent import Player
import pandas as pd

logging.basicConfig(level=logging.INFO)

def compute_individual_banish_rate(row):
    dp = row.get("discussion_participation", 0)
    bd = row.get("banished_in_discussion", 0)
    if dp > 0:
        return bd / dp
    else:
        return None

def main():
    # Create a game instance (with discussion enabled)
    game = Game(discussion=True)

    # Create players.
    players = [
        Player("Liam",   killer=False, preprompt="prompt_2", agent="gpt-4o-mini-2024-07-18"),
        Player("Oliver", killer=False, preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
        Player("James",  killer=False, preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
        Player("Ezra",   killer=False, preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
        Player("Asher",  killer=False, preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
        Player("Mason",  killer=True, preprompt="prompt_1", agent="gpt-4o-mini-2024-07-18"),
    ]

    game.load_players(players)
    results = game.play()  # List of evaluation dictionaries

    # Build a DataFrame and compute individual banish_rate.
    df = pd.DataFrame(results)
    df["banish_rate"] = df.apply(compute_individual_banish_rate, axis=1)

    # Section 1: Player Metrics Table
    selected_keys = [
        "agent", "killer", "preprompt", "num_turns", "banished",
        "escaped", "killed", "vote_rate_for_killer", "vote_rate_for_self",
        "discussion_participation", "banish_rate"
    ]
    df_subset = df.set_index("name")[selected_keys].transpose()
    # (Optional) Reorder columns if desired. Here we check a desired order list:
    desired_order = ["Mira", "Dave", "Tom", "Archie"]
    existing_names = [name for name in desired_order if name in df_subset.columns]
    if existing_names:
        df_subset = df_subset[existing_names]

    # Section 2: Overall Summary
    summary_records = []
    for pre in ["prompt_1", "prompt_2"]:
        subset = df[df["preprompt"] == pre]
        count = len(subset)
        banished_count = subset["banished"].sum()
        avg_vote_rate = subset["vote_rate_for_killer"].mean() if "vote_rate_for_killer" in subset.columns and not subset["vote_rate_for_killer"].isnull().all() else None
        avg_self_vote_rate = subset["vote_rate_for_self"].mean() if "vote_rate_for_self" in subset.columns and not subset["vote_rate_for_self"].isnull().all() else None
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

    # Section 3: Prompts Used (print once at the end)
    # Specify the order and keys you want to print
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
