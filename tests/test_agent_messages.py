"""Image ownership remains stable when repair messages follow an observation."""
import unittest

from agent.messages import attach_image


class ImageMessageTests(unittest.TestCase):
    def test_attach_to_observation_without_mutating_history(self):
        messages = [
            {"role": "system", "content": "JSON protocol"},
            {"role": "user", "content": "task and observation"},
            {"role": "assistant", "content": "bad JSON"},
            {"role": "user", "content": "repair JSON"},
        ]
        for _ in range(2):
            prepared = attach_image(messages, 1)
            self.assertEqual(prepared[1]["content"], "[IMG]\ntask and observation")
            self.assertEqual(prepared[-1]["content"], "repair JSON")
            self.assertEqual(messages[1]["content"], "task and observation")
        self.assertEqual(attach_image(messages)[-1]["content"], "[IMG]\nrepair JSON")
        for index in (-1, 4, True, 0, 2):
            with self.subTest(index=index), self.assertRaises(ValueError):
                attach_image(messages, index)
