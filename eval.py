# eval.py
import os
import sys
import pandas as pd
import time
import logging
import argparse
import warnings

from environment import Game
from agent import Player

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO)  # set global logging

"""
Script for running multiple games as a batch. 
"""

def run_job(num_games, num_players, impostor_agent, innocent_agent, discussion, start_location, eval_cols):
    """
    Runs a number of games with the given specifications. 
    Returns a dictionary of eval results with a row for each player.
    """
    eval_dict = {col: [] for col in eval_cols}

    for i in range(1, num_games+1):
        start_time = time.time()
        try:
            game = Game(discussion=discussion)

            # If you want a custom start location approach, might do it in environment
            # For now, we just call load_random_players
            game.load_random_players(num_players, impostor_agent, innocent_agent)

            results = game.play()
            end_time = time.time()
            runtime = end_time - start_time

            # Collate
            for player_res in results:
                for k in list(eval_dict.keys())[4:]:
                    eval_dict[k].append(player_res.get(k, ""))
            # Add game-level columns
            eval_dict["game_num"].extend([i]*num_players)
            eval_dict["runtime"].extend([runtime]*num_players)
            eval_dict["num_players"].extend([num_players]*num_players)
            eval_dict["discussion"].extend([discussion]*num_players)

        except Exception as e:
            logging.error("Error in run_job loop: %s. Sleeping 30s...", e)
            time.sleep(30)
            continue

    return eval_dict

def get_save_path():
    """
    Returns a new file path in 'results/' with an integer filename.
    """
    save_dir = 'results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_count = sum(os.path.isfile(os.path.join(save_dir, name)) for name in os.listdir(save_dir))
    return os.path.join(save_dir, f"{file_count}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process job info.')
    parser.add_argument('--job_number', type=int, required=True,
                        help="Which CSV file in /jobs to run.")
    args = parser.parse_args()
    job_number = args.job_number

    schedule_path = f"jobs/{job_number}.csv"
    if not os.path.exists(schedule_path):
        print(f"No schedule file found: {schedule_path}")
        sys.exit(1)

    schedule = pd.read_csv(schedule_path)
    save_path = get_save_path()

    # Structure
    results_cols = [
        "game_num", "runtime", "num_players", "discussion",
        "name", "agent", "killer", "num_turns", "banished",
        "killed", "escaped", "num_killed", "num_escaped",
        "duplicate_search_rate", "vote_rate_for_self", "vote_rate_for_killer",
        "witness_vote_rate_for_killer", "non_witness_vote_rate_for_killer",
        "story", "actions", "votes", "witness_during_vote"
    ]
    overall = {col: [] for col in results_cols}

    for idx, row in schedule.iterrows():
        runs = run_job(
            num_games=int(row['num_games']),
            num_players=int(row['num_players']),
            impostor_agent=str(row['impostor_agent']),
            innocent_agent=str(row['innocent_agent']),
            discussion=bool(row['discussion']),
            start_location=str(row['start_location']),
            eval_cols=results_cols
        )
        for k, v in runs.items():
            overall[k].extend(v)

        # Save after each job to reduce risk of data loss
        df = pd.DataFrame(overall)
        df.to_csv(save_path, index=False)
    print(f"Saved results to {save_path}")
 
