# test_agent.py
import unittest
from agent import Player
from constants import SEARCH_PREFIX

class TestPlayerMethods(unittest.TestCase):
    def setUp(self):
        self.player = Player(
            name="Alice",
            killer=False,
            preprompt="rules",
            agent="random",  # or "cli"/"gpt-3.5" if desired
            start_location="Kitchen"
        )
    
    def test_decode_action(self):
        action_prompt = """
        Possible Actions:
        1. Search the fridge
        2. Search the cabinets
        3. Go to the Hallway
        """
        # Suppose the user picks "2" => "Search the cabinets"
        action_text = self.player._decode_action(action_prompt, 2)
        self.assertEqual(action_text, "Search the cabinets")

    def test_decode_vote(self):
        vote_prompt = """
        1. Bob
        2. Jim
        3. Mira
        """
        vote_target = self.player._decode_vote(vote_prompt, 3)
        self.assertEqual(vote_target, "Mira")

    def test_final_eval(self):
        self.player.finalize_eval(killer_name="Bob")
        # Basic check that no error is raised and eval dict is updated
        self.assertIn("actions", self.player.eval)
        self.assertIn("votes", self.player.eval)
        # For a non-killer, we should have 'escaped' or 'killed' keys
        self.assertIn("escaped", self.player.eval)
        self.assertFalse(self.player.eval["escaped"])

    def test_search_duplicates(self):
        # The player is not a killer
        self.player.actions = [
            f"{SEARCH_PREFIX}the fridge",
            f"{SEARCH_PREFIX}the fridge",
            f"{SEARCH_PREFIX}the cabinets"
        ]
        self.player.finalize_eval(killer_name="Bob")
        dup_rate = self.player.eval.get("duplicate_search_rate", 0)
        # 2 of 3 searches? Actually 1 duplicate out of 3 => 1/3
        self.assertAlmostEqual(dup_rate, 1/3, places=5)

if __name__ == '__main__':
    unittest.main()
