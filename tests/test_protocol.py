"""Robust JSON extraction for model decisions (Qwen/thinking models often wrap JSON)."""
import unittest

from agent.protocol import parse_decision


class RobustParseTests(unittest.TestCase):
    def test_fenced_block_with_surrounding_text(self):
        text = 'Plan:\n```json\n{"tool": "takeoff"}\n```\nDone.'
        self.assertEqual(parse_decision(text), {"tool": "takeoff"})

    def test_object_inside_prose(self):
        text = 'I will take off now. {"tool": "takeoff"} end'
        self.assertEqual(parse_decision(text), {"tool": "takeoff"})

    def test_single_element_array(self):
        self.assertEqual(parse_decision('[{"final":"ok"}]'), {"final": "ok"})

    def test_double_encoded_json_string(self):
        self.assertEqual(parse_decision(r'"{\"tool\":\"takeoff\"}"'), {"tool": "takeoff"})

    def test_non_object_values_rejected(self):
        for value in ("[]", '[{"tool":"takeoff"},{"tool":"land"}]', "null", "42", '"takeoff"'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_decision(value)


if __name__ == "__main__":
    unittest.main()
