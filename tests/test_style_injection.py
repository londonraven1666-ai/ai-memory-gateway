import unittest

from style_injection import inject_style_into_current_user_message, parse_style, parse_style_list


class StyleInjectionTests(unittest.TestCase):
    def test_injects_into_last_user_string_without_mutating_input(self):
        messages = [
            {"role": "system", "content": "stable"},
            {"role": "user", "content": "hello"},
        ]

        result = inject_style_into_current_user_message(messages, "Reply casually.")

        self.assertEqual(messages[1]["content"], "hello")
        self.assertEqual(
            result[1]["content"],
            "hello\n\n<style_instructions>\nReply casually.\n</style_instructions>",
        )
        self.assertIsNot(result, messages)
        self.assertIsNot(result[1], messages[1])

    def test_injects_into_multimodal_text_without_mutating_blocks(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,x"}},
                    {"type": "text", "text": "describe this"},
                ],
            }
        ]

        result = inject_style_into_current_user_message(messages, "Be concise.")

        self.assertEqual(messages[0]["content"][1]["text"], "describe this")
        self.assertTrue(result[0]["content"][1]["text"].endswith("</style_instructions>"))

    def test_parsers_reject_invalid_values(self):
        self.assertEqual(parse_style_list('["Daily", "Daily", 3]'), ["Daily"])
        self.assertIsNone(parse_style('{"title":"Daily"}'))


if __name__ == "__main__":
    unittest.main()
