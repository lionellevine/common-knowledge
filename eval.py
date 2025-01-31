# eval.py

import os
import sys
import random
import time
import logging
import argparse
import warnings
import pandas as pd

from environment import Game
from agent import Player

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO)


def run_job(num_games, eval_cols):
    """
    Runs `num_games`, each time using exactly 4 named players [Jacob,Kai,Archie,Luca].
    - Randomly pick 1 as killer
    - Randomly pick 1 as 'rules_alt' user
    - The rest are 'rules' + not killer
    After each game, append a short "comparison" row for alt vs. normal in terms of guess-killer rate, etc.
    """
    eval_dict = {col: [] for col in eval_cols}

    # We'll accumulate alt vs. rules data for final summary
    # For each player row, we know "preprompt" => either "rules_alt" or "rules"
    # We'll track their total "vote_rate_for_killer", number of players, etc.
    # We'll do that in 'overall_summary' mapping from "rules_alt"|"rules" => a dict of sums
    overall_summary = {
        "rules_alt": {"count": 0, "sum_vote_for_killer": 0.0},
        "rules":     {"count": 0, "sum_vote_for_killer": 0.0},
    }

    for game_idx in range(1, num_games + 1):
        start_time = time.time()

        # 1) Randomly assign the roles
        name_list = ["Jacob", "Kai", "Archie", "Luca"]
        random.shuffle(name_list)  # shuffle the 4 names
        killer_name = random.choice(name_list)  # pick 1 killer
        alt_name = random.choice(name_list)     # pick 1 to have rules_alt (could be the same or not, up to you)

        # 2) Build the 4 players
        players = []
        for nm in name_list:
            is_killer = (nm == killer_name)
            pp = "rules_alt" if nm == alt_name else "rules"
            p = Player(
                name=nm,
                killer=is_killer,
                preprompt=pp,
                agent="gpt-4o-mini-2024-07-18"  # or "gpt" or whatever
            )
            players.append(p)

        # 3) Create the Game, load the 4 players
        game = Game(discussion=True)
        game.load_players(players)

        # 4) Insert a "GameHeader" row describing the players
        insert_game_header_row(eval_dict, game_idx, name_list, alt_name, killer_name)

        # 5) Play the game
        results = game.play()
        end_time = time.time()
        runtime = end_time - start_time

        # 6) Insert normal player rows into the CSV
        for player_res in results:
            insert_player_result_row(eval_dict, game_idx, runtime, 4, True, player_res)

            # We'll also update overall_summary
            pre = player_res.get("preprompt", "rules")  # either 'rules_alt' or 'rules'
            # just in case, if it's something else, group as 'rules'
            if pre not in ("rules_alt", "rules"):
                pre = "rules"

            # vote_rate_for_killer is how often they guessed the killer.
            # We'll add to the sums
            overall_summary[pre]["count"] += 1
            overall_summary[pre]["sum_vote_for_killer"] += player_res.get("vote_rate_for_killer", 0.0)

        # 7) Insert a Stats row from the environment code (like killed=, banished=, etc.)
        #    We can do something similar to your existing code: e.g. sum up banished, killed, etc.
        banished_count = sum(r.get("banished", 0) for r in results)
        killed_count   = sum(r.get("killed", 0) for r in results)
        escaped_count  = sum(r.get("escaped", 0) for r in results)

        stats_text = (
            f"======================================= GAME #{game_idx} STATS =======================================\n"
            f"RuntimeSec={runtime:.2f}\n"
            f"Banished={banished_count}\n"
            f"Killed={killed_count}\n"
            f"Escaped={escaped_count}"
        )
        insert_special_row(eval_dict, stats_text)

        # 8) Insert a "Comparison" row for alt vs. normal in this single game
        alt_vfk   = 0.0
        alt_count = 0
        rules_vfk   = 0.0
        rules_count = 0

        for r in results:
            if r["preprompt"] == "rules_alt":
                alt_vfk   += r.get("vote_rate_for_killer", 0.0)
                alt_count += 1
            else:
                rules_vfk   += r.get("vote_rate_for_killer", 0.0)
                rules_count += 1

        if alt_count > 0:
            alt_avg = alt_vfk / alt_count
        else:
            alt_avg = 0.0

        if rules_count > 0:
            rules_avg = rules_vfk / rules_count
        else:
            rules_avg = 0.0

        comp_text = (
            f"======================================= COMPARISON FOR GAME #{game_idx} =======================================\n"
            f" alt_avg_vote_for_killer={alt_avg:.2f}\n"
            f" rules_avg_vote_for_killer={rules_avg:.2f}\n"
            f" (Killer was {killer_name}, alt was {alt_name})"
        )
        insert_special_row(eval_dict, comp_text)

        # 9) Insert an end-of-run row
        end_run_text = f"----- END OF RUN #{game_idx} ----- (runtime={runtime:.2f}s)"
        insert_special_row(eval_dict, end_run_text)

    return eval_dict, overall_summary


def insert_game_header_row(eval_dict, game_index, name_list, alt_name, killer_name):
    """
    Insert a row describing which 4 players are in this game, who is the killer, who is alt, etc.
    """
    line1 = f"============================================== Game #{game_index} =============================================="
    line2 = f"NumPlayers=4, Discussion=True"
    lines_players = []
    for nm in name_list:
        is_k = (nm == killer_name)
        is_alt = (nm == alt_name)
        lines_players.append(
            f"  - {nm}: killer={is_k}, alt={is_alt}"
        )
    player_list_block = "\n".join(lines_players)
    final_text = f"{line1}\n{line2}\nPlayers:\n{player_list_block}"

    insert_special_row(eval_dict, final_text)


def insert_player_result_row(eval_dict, game_idx, runtime, num_players, discussion, player_res):
    """
    Creates a multiline "key: value" text in 'PlayerData' column for normal rows.
    """

    data_str = build_player_data_str(game_idx, runtime, num_players, discussion, player_res)
    eval_dict["RowNotes"].append("")     # no special note, it's a normal row
    eval_dict["PlayerData"].append(data_str)

    # FullStoryLog, etc.
    eval_dict["FullStoryLog"].append(player_res.get("story", ""))
    eval_dict["ActionsTaken"].append(player_res.get("actions", []))
    eval_dict["VotesCast"].append(player_res.get("votes", []))
    eval_dict["WitnessDuringVote"].append(player_res.get("witness_during_vote", []))


def build_player_data_str(game_idx, runtime, num_players, discussion, player_res):
    """Return multiline 'key: value' block for a single player's final result."""
    lines = []
    lines.append(f"GameNumber: {game_idx}")
    lines.append(f"RuntimeSec: {runtime:.2f}")
    lines.append(f"NumPlayers: {num_players}")
    lines.append(f"DiscussionOn: {discussion}")
    lines.append(f"PlayerName: {player_res.get('name','')}")
    lines.append(f"AgentType: {player_res.get('agent','')}")
    lines.append(f"Preprompt: {player_res.get('preprompt','')}")
    lines.append(f"IsKiller: {player_res.get('killer',False)}")
    lines.append(f"WasBanished: {player_res.get('banished',False)}")
    lines.append(f"WasKilled: {player_res.get('killed',False)}")
    lines.append(f"DidEscape: {player_res.get('escaped',False)}")
    lines.append(f"NumTurns: {player_res.get('num_turns',0)}")
    lines.append(f"NumKilled: {player_res.get('num_killed',0)}")
    lines.append(f"NumEscaped: {player_res.get('num_escaped',0)}")
    lines.append(f"InvalidVotesForEliminated: {player_res.get('invalid_votes_for_eliminated',0)}")
    mk = player_res.get("multiple_killers", [])
    if isinstance(mk, list):
        mk_str = ",".join(mk)
    else:
        mk_str = str(mk)
    lines.append(f"MultipleKillers: {mk_str}")
    lines.append(f"DuplicateSearchRate: {player_res.get('duplicate_search_rate',0.0)}")
    lines.append(f"VoteRateForSelf: {player_res.get('vote_rate_for_self',0.0)}")
    lines.append(f"VoteRateForKiller: {player_res.get('vote_rate_for_killer',0.0)}")
    lines.append(f"WitnessVoteRateForKiller: {player_res.get('witness_vote_rate_for_killer',0.0)}")
    lines.append(f"NonWitnessVoteRateForKiller: {player_res.get('non_witness_vote_rate_for_killer',0.0)}")
    return "\n".join(lines)


def insert_special_row(eval_dict, text_value):
    """Inserts a row with multiline text in RowNotes. PlayerData is empty, other columns blank."""
    eval_dict["RowNotes"].append(text_value)
    eval_dict["PlayerData"].append("")

    eval_dict["FullStoryLog"].append("")
    eval_dict["ActionsTaken"].append([])
    eval_dict["VotesCast"].append([])
    eval_dict["WitnessDuringVote"].append([])


def get_save_path():
    """Creates unique path in 'results/' directory."""
    save_dir = 'results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_count = sum(os.path.isfile(os.path.join(save_dir, f)) for f in os.listdir(save_dir))
    return os.path.join(save_dir, f"{file_count}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_number', type=int, required=True)
    parser.add_argument('--num_games', type=int, default=3, help="How many games in this job")
    args = parser.parse_args()

    # We'll ignore 'job_number' except for reading a schedule if you want. For simplicity, let's just run num_games:
    job_number = args.job_number
    num_games  = args.num_games

    # Our final columns:
    results_cols = [
        "RowNotes",       # multiline text for headers/stats/special
        "PlayerData",     # multiline "key: value" for normal players
        "FullStoryLog",
        "ActionsTaken",
        "VotesCast",
        "WitnessDuringVote",
    ]
    # For demonstration, we won't parse a CSV schedule. We'll just run `num_games`.
    # If you prefer, you can read some "jobs/<job_number>.csv" as well.

    eval_dict, overall_summary = run_job(num_games, results_cols)

    # Convert to DataFrame
    df = pd.DataFrame(eval_dict, columns=results_cols)

    # Save partial
    save_path = get_save_path()
    df.to_csv(save_path, index=False)
    print(f"\nSaved intermediate CSV to {save_path}\n")

    # === After all games, produce final summary comparing alt vs. rules overall ===
    alt_count = overall_summary["rules_alt"]["count"]
    alt_vsum  = overall_summary["rules_alt"]["sum_vote_for_killer"]
    rules_count = overall_summary["rules"]["count"]
    rules_vsum  = overall_summary["rules"]["sum_vote_for_killer"]

    alt_avg   = alt_vsum / alt_count   if alt_count   > 0 else 0.0
    rules_avg = rules_vsum / rules_count if rules_count > 0 else 0.0

    final_text = (
        "======================================= FINAL SUMMARY: 'rules_alt' vs. 'rules' =======================================\n"
        f"  alt_count={alt_count}, alt_avg_vote_for_killer={alt_avg:.2f}\n"
        f"  rules_count={rules_count}, rules_avg_vote_for_killer={rules_avg:.2f}\n"
        "-----------------------------------"
    )

    # Insert as final row
    df.loc[len(df)] = [final_text, "", "", [], [], []]

    # Overwrite CSV
    df.to_csv(save_path, index=False)
    print("Appended final summary at bottom.")
    print(f"Final CSV saved to {save_path}")
