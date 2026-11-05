"""Image ownership remains stable when repair messages follow an observation."""
import unittest

from agent.messages import attach_image, prepare_hf_images
from PIL import Image


class ImageMessageTests(unittest.TestCase):
    def test_historical_images_follow_message_order_and_current_retry_binding(self):
        red, blue, green, gray = [Image.new("RGB", (2, 2), color)
                                  for color in ("red", "blue", "green", "gray")]
        messages = [
            {"role": "system", "content": "JSON"},
            {"role": "user", "content": "historical observation"},
            {"role": "assistant", "content": "tool"},
            {"role": "user", "content": "current observation"},
            {"role": "assistant", "content": "invalid JSON"},
            {"role": "user", "content": "repair"},
        ]
        rendered, images = prepare_hf_images(
            messages, green, 3, {"process_rgb_000000": gray},
            {1: {"rgb": red, "depth": blue}})
        self.assertEqual(images, [red, blue, green, gray])
        self.assertEqual(rendered[1]["content"].count("[IMG]"), 2)
        self.assertEqual(rendered[3]["content"].count("[IMG]"), 2)
        self.assertEqual(rendered[-1]["content"], "repair")
        self.assertEqual(messages[1]["content"], "historical observation")
        with self.assertRaises(ValueError):
            prepare_hf_images(messages, message_images={2: {"rgb": red}})

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
