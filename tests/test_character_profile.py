"""Tests for :mod:`sml_extractor.character_profile`."""

import unittest

from sml_extractor import character_profile as cp
from sml_extractor.llm_client import LlmError


BOOK_TXT = "\n".join(
    [
        "Пётр Иванов вошёл в комнату и осмотрелся. Он был суровым человеком.",
        "[Пётр Иванов] Я не намерен работать на этого человека. Работа и дело важнее всего. [/]",
        "[Мария] Пётр, почему вы так суровы? Люди хотят покоя и разговоров. [/]",
        "Мария Петрова улыбнулась. Она любила спокойные разговоры и порядок.",
        "[Пётр Иванов] Служба и деньги — вот что важно. Дело требует службы. [/]",
        "[Мария] Покой, разговоры и здоровье полезны людям. [/]",
    ]
)

TOKENS = [
    {"token_ID_within_document": 0, "word": "Довольно", "paragraph_ID": 1},
    {"token_ID_within_document": 1, "word": "трепаться", "paragraph_ID": 1},
    {"token_ID_within_document": 2, "word": "Пора", "paragraph_ID": 1},
    {"token_ID_within_document": 3, "word": "работать", "paragraph_ID": 1},
]
QUOTES = [{"quote_start": 0, "quote_end": 1, "char_id": 5}]
BOOK_DATA = {
    "characters": [{"id": 5, "mentions": {"proper": [{"n": "Пётр"}], "common": []}}]
}

PAYLOAD = {
    "descriptions": [
        {"character": "ПётрИванов", "description": "Суровый работяга, говорит о деле и деньгах."},
        {"character": "МарияПетрова", "description": "Мягкая женщина, любит покой и разговоры."},
    ]
}


def make_characters():
    return [
        {
            "normalized_name": "Narrator",
            "inferred_gender": "unknown",
            "inferred_age_category": "adult",
        },
        {
            "normalized_name": "ПётрИванов",
            "inferred_gender": "male",
            "inferred_age_category": "adult",
        },
        {
            "normalized_name": "МарияПетрова",
            "inferred_gender": "female",
            "inferred_age_category": "elder",
        },
    ]


def by_name(characters):
    return {character["normalized_name"]: character for character in characters}


class FakeClient:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    def chat_json(self, system, user, **kwargs):
        self.calls.append({"system": system, "user": user, "kwargs": kwargs})
        if self.error is not None:
            raise self.error
        return self.payload


class EvidenceTests(unittest.TestCase):
    def test_quotes_and_mentions_from_book_txt(self):
        characters = make_characters()
        evidence = cp.build_evidence({"book_txt": BOOK_TXT}, characters)

        petr = evidence["ПётрИванов"]
        self.assertEqual(petr["quote_count"], 2)
        self.assertIn("намерен", petr["quotes"][0])
        self.assertTrue(petr["mentions"])
        self.assertIn("Пётр Иванов", petr["mentions"][0])

        # the dialogue labels only the first name, the full name still matches
        maria = evidence["МарияПетрова"]
        self.assertEqual(maria["quote_count"], 2)
        self.assertIn("покоя", maria["quotes"][0])

        self.assertEqual(evidence["Narrator"]["quotes"], [])
        self.assertTrue(evidence["Narrator"]["mentions"])

    def test_quotes_from_tokens(self):
        evidence = cp.build_evidence(
            {"tokens": TOKENS, "quotes": QUOTES, "book_data": BOOK_DATA},
            [
                {
                    "normalized_name": "Пётр",
                    "inferred_gender": "male",
                    "inferred_age_category": "adult",
                }
            ],
        )
        self.assertEqual(evidence["Пётр"]["quotes"], ["Довольно трепаться"])
        self.assertEqual(evidence["Пётр"]["quote_count"], 1)

    def test_no_data_gives_empty_evidence(self):
        evidence = cp.build_evidence(None, make_characters())
        self.assertEqual(
            set(evidence), {"Narrator", "ПётрИванов", "МарияПетрова"}
        )
        for data in evidence.values():
            self.assertEqual(data["quotes"], [])
            self.assertEqual(data["mentions"], [])


class HeuristicTests(unittest.TestCase):
    def test_gender_age_phrase_and_words(self):
        character = {
            "normalized_name": "ПётрИванов",
            "inferred_gender": "male",
            "inferred_age_category": "adult",
        }
        text = cp.heuristic_description(
            character, {"quote_count": 7}, language="rus", words=["дело", "служба"]
        )
        self.assertTrue(text.startswith("Взрослый мужчина"))
        self.assertIn("7 реплик", text)
        self.assertIn("дело, служба", text)

    def test_english_phrases(self):
        character = {
            "normalized_name": "MariaPetrova",
            "inferred_gender": "female",
            "inferred_age_category": "elder",
        }
        text = cp.heuristic_description(character, language="eng")
        self.assertTrue(text.startswith("An elderly woman"))
        self.assertIn("0 lines", text)

    def test_narrator_phrases(self):
        self.assertIn(
            "рассказчик",
            cp.heuristic_description({"normalized_name": "Narrator"}, language="rus"),
        )
        self.assertTrue(
            cp.heuristic_description(
                {"normalized_name": "Narrator"}, language="eng"
            ).startswith("The narrator")
        )


class CharacteristicWordsTests(unittest.TestCase):
    def test_shared_words_are_dropped(self):
        evidence = {
            "A": {"quotes": ["Работа и деньги важнее всего"], "mentions": ["человек человек"]},
            "B": {"quotes": ["Люди хотят покоя"], "mentions": ["человек"]},
            "C": {"quotes": ["Служба требует порядка"], "mentions": ["человек"]},
        }
        words = cp.characteristic_words(evidence)
        self.assertIn("работа", words["A"])
        self.assertNotIn("человек", words["A"])
        self.assertNotIn("человек", words["B"])

    def test_stopwords_are_dropped(self):
        words = cp.characteristic_words({"A": {"quotes": ["этот очень просто сказал"]}})
        self.assertEqual(words["A"], [])

    def test_limit(self):
        evidence = {"A": {"quotes": ["работа деньги служба порядок дело"]}}
        words = cp.characteristic_words(evidence, limit=2)
        self.assertEqual(len(words["A"]), 2)


class DescribeTests(unittest.TestCase):
    def test_llm_descriptions_replace_the_heuristic(self):
        characters = make_characters()
        client = FakeClient(payload=PAYLOAD)
        cp.describe_characters(characters, {"book_txt": BOOK_TXT}, client=client)

        found = by_name(characters)
        self.assertEqual(found["ПётрИванов"]["description_source"], "llm")
        self.assertEqual(
            found["ПётрИванов"]["description"],
            "Суровый работяга, говорит о деле и деньгах.",
        )
        self.assertEqual(found["МарияПетрова"]["description_source"], "llm")
        # the narrator is missing from the answer, so the heuristic stays
        self.assertEqual(found["Narrator"]["description_source"], "heuristic")
        self.assertEqual(len(client.calls), 1)
        self.assertIn("ПётрИванов", client.calls[0]["user"])

    def test_llm_error_falls_back(self):
        characters = make_characters()
        client = FakeClient(error=LlmError("no model"))
        cp.describe_characters(characters, {"book_txt": BOOK_TXT}, client=client)
        self.assertTrue(
            all(c["description_source"] == "heuristic" for c in characters)
        )
        self.assertTrue(all(c["description"] for c in characters))

    def test_generic_error_falls_back(self):
        characters = make_characters()
        client = FakeClient(error=RuntimeError("boom"))
        cp.describe_characters(characters, {"book_txt": BOOK_TXT}, client=client)
        self.assertTrue(
            all(c["description_source"] == "heuristic" for c in characters)
        )

    def test_partial_answer_keeps_the_heuristic(self):
        characters = make_characters()
        client = FakeClient(
            payload={
                "descriptions": [
                    {"character": "ПётрИванов", "description": "Только он."}
                ]
            }
        )
        cp.describe_characters(characters, {"book_txt": BOOK_TXT}, client=client)
        found = by_name(characters)
        self.assertEqual(found["ПётрИванов"]["description"], "Только он.")
        self.assertEqual(found["ПётрИванов"]["description_source"], "llm")
        self.assertEqual(found["МарияПетрова"]["description_source"], "heuristic")
        self.assertEqual(found["Narrator"]["description_source"], "heuristic")

    def test_user_description_is_kept(self):
        characters = make_characters()
        characters[1]["description"] = "Голос из моих заметок"
        client = FakeClient(payload=PAYLOAD)
        cp.describe_characters(characters, {"book_txt": BOOK_TXT}, client=client)

        self.assertEqual(characters[1]["description"], "Голос из моих заметок")
        self.assertEqual(characters[1]["description_source"], "user")
        self.assertNotIn("Голос из моих заметок", client.calls[0]["user"])

    def test_use_llm_false_keeps_the_heuristic(self):
        characters = make_characters()
        client = FakeClient(error=AssertionError("must not be called"))
        cp.describe_characters(
            characters, {"book_txt": BOOK_TXT}, client=client, use_llm=False
        )
        self.assertEqual(client.calls, [])
        self.assertTrue(
            all(c["description_source"] == "heuristic" for c in characters)
        )

    def test_limit_restricts_the_llm_pass(self):
        characters = make_characters()
        client = FakeClient(payload=PAYLOAD)
        cp.describe_characters(
            characters, {"book_txt": BOOK_TXT}, client=client, limit=1
        )
        # only the first character with evidence of its own reaches the model
        self.assertEqual(len(client.calls), 1)
        self.assertIn("ПётрИванов", client.calls[0]["user"])
        self.assertNotIn("МарияПетрова", client.calls[0]["user"])

    def test_thin_evidence_keeps_the_heuristic(self):
        characters = make_characters()
        book = "\n".join(
            [
                "[Пётр Иванов] Работа и дело важнее всего. [/]",
                "Пётр Иванов кивнул и вышел из комнаты.",
            ]
        )
        client = FakeClient(payload=PAYLOAD)
        cp.describe_characters(characters, {"book_txt": book}, client=client)

        found = by_name(characters)
        # nobody said a word about her and the narrator is never cast from the text
        self.assertEqual(found["МарияПетрова"]["description_source"], "heuristic")
        self.assertEqual(found["Narrator"]["description_source"], "heuristic")
        for call in client.calls:
            self.assertNotIn("Narrator", call["user"])
            self.assertNotIn("МарияПетрова", call["user"])

    def test_progress_is_reported(self):
        characters = make_characters()
        seen = []
        cp.describe_characters(
            characters,
            {"book_txt": BOOK_TXT},
            client=FakeClient(payload=PAYLOAD),
            progress=lambda fraction, desc=None: seen.append((fraction, desc)),
        )
        self.assertTrue(seen)
        self.assertEqual(seen[-1][0], 1.0)


class ParseBatchTests(unittest.TestCase):
    def test_keyed_dict(self):
        self.assertEqual(
            cp._parse_batch({"1": "A", "2": "B"}, 2, ["Пётр", "Мария"]), ["A", "B"]
        )

    def test_plain_list(self):
        self.assertEqual(cp._parse_batch(["A", "B"], 2), ["A", "B"])

    def test_name_lookup_ignores_spacing_and_case(self):
        self.assertEqual(cp._parse_batch({"Пётр Иванов": "A"}, 1, ["ПётрИванов"]), ["A"])

    def test_short_form_matches_the_full_name(self):
        payload = {"descriptions": [{"name": "Мария", "description": "B"}]}
        self.assertEqual(cp._parse_batch(payload, 1, ["МарияПетрова"]), ["B"])

    def test_missing_entries_are_none(self):
        self.assertEqual(
            cp._parse_batch({"descriptions": []}, 2, ["A", "B"]), [None, None]
        )


if __name__ == "__main__":
    unittest.main()
