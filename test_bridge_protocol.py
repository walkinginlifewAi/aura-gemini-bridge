import unittest

from app import summarize_message_chain


class MessageChainSummaryTest(unittest.TestCase):
    def test_summarizes_tool_identifiers_without_message_content(self):
        summary = summarize_message_chain([
            {"role": "user", "content": "sensitive text that must not be logged"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_123", "function": {"name": "tool"}}],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "sensitive result"},
        ])

        self.assertEqual(summary[0]["role"], "user")
        self.assertEqual(summary[1]["assistant_tool_calls"], 1)
        self.assertEqual(summary[1]["assistant_tool_ids_present"], 1)
        self.assertTrue(summary[2]["tool_result_id_present"])
        self.assertNotIn("sensitive text", str(summary))
        self.assertNotIn("sensitive result", str(summary))


if __name__ == "__main__":
    unittest.main()
