"""Tests for explicit key-character parsing and merging."""

import unittest
from unittest.mock import patch

from sml_extractor.character_input import (
    _match_score,
    merge_key_characters,
    parse_key_characters,
    parse_key_characters_heuristic,
)
from sml_extractor.llm_client import LlmError


SAMPLE_NOTES = """Примарх и легион «Лунные Волки» (позже — Сыны Хоруса)
• Хорус Луперкаль — Воитель (Warmaster), любимый сын Императора и предводитель легиона.
• Гарвель Локен — капитан 10-й роты (главный герой и рассказчик).
• Эуфратия Киилер — официальный летописец, фотограф (ключевая фигура).
Первый капитан — Севатар.
- Далее, Саул Тарвиль
"""


class HeuristicParsingTests(unittest.TestCase):
    def test_parses_bullets_and_skips_headings(self):
        records = parse_key_characters_heuristic(
            'Примарх и легион «Лунные Волки»\n'
            '• Хорус Луперкаль — Воитель, любимый сын Императора.\n'
            '• Эуфратия Киилер — официальный летописец, фотограф.'
        )
        names = [record['name'] for record in records]
        self.assertEqual(names, ['Хорус Луперкаль', 'Эуфратия Киилер'])
        self.assertEqual(records[0]['gender'], 'male')
        self.assertEqual(records[1]['gender'], 'female')
        self.assertIn('Воитель', records[0]['description'])

    def test_reversed_role_to_name_and_discourse_marker(self):
        records = parse_key_characters_heuristic(
            'Первый капитан — Севатар.\nДалее, Саул Тарвиль'
        )
        names = [record['name'] for record in records]
        self.assertEqual(names, ['Севатар', 'Саул Тарвиль'])
        self.assertIn('Первый капитан', records[0]['description'])

    def test_full_sample_keeps_every_named_person(self):
        names = [record['name'] for record in parse_key_characters_heuristic(
            'Примарх и легион «Лунные Волки»\n'
            '• Хорус Луперкаль — Воитель (Warmaster), предводитель легиона.\n'
            '• Малогарст по прозвищу «Кривой» — советник Хоруса.\n'
            '• Гарвель Локен — капитан 10-й роты.\n'
            'Первый капитан — Севатар.\n'
            '- Далее, Саул Тарвиль'
        )]
        self.assertEqual(len(names), 5)
        self.assertIn('Севатар', names)
        self.assertIn('Саул Тарвиль', names)

    def test_finish_reasons_and_headers(self):
        records = parse_key_characters_heuristic(
            'Легион «Лунные Волки» (позже — Сыны Хоруса):\n'
            '• Гарвель Локен — капитан 10-й роты, главный герой.\n'
            '• Локен — см. выше.'
        )
        self.assertEqual([r['name'] for r in records], ['Гарвель Локен'])

    def test_empty_text(self):
        self.assertEqual(parse_key_characters_heuristic(''), [])


class LlmParsingTests(unittest.TestCase):
    class FakeClient:
        def __init__(self, payload=None, error=None):
            self.payload = payload
            self.error = error
            self.calls = []

        def chat_json(self, system, user, **kwargs):
            self.calls.append((system, user))
            if self.error:
                raise self.error
            return self.payload

    def test_llm_answer_is_normalized(self):
        client = self.FakeClient({'characters': [
            {'name': 'Хорус Луперкаль', 'aliases': ['Хорус', 'Воитель'],
             'gender': 'Male', 'age': 'adult', 'description': 'примарх'},
            {'name': 'Эуфратия', 'gender': 'нечто', 'age': 'вечность',
             'description': 'летописец'},
            {'name': '  '},
        ]})
        records = parse_key_characters('anything', client=client)
        self.assertEqual([r['name'] for r in records], ['Хорус Луперкаль', 'Эуфратия'])
        self.assertEqual(records[0]['gender'], 'male')
        self.assertEqual(records[0]['aliases'], ['Хорус', 'Воитель'])
        self.assertEqual(records[1]['gender'], 'unknown')
        self.assertEqual(records[1]['age'], 'unknown')

    def test_llm_failure_falls_back_to_heuristic(self):
        client = self.FakeClient(error=LlmError('offline'))
        records = parse_key_characters('Хорус — примарх', client=client)
        self.assertEqual([r['name'] for r in records], ['Хорус'])

    def test_empty_llm_answer_falls_back(self):
        client = self.FakeClient({'characters': []})
        records = parse_key_characters('Хорус — примарх', client=client)
        self.assertEqual([r['name'] for r in records], ['Хорус'])

    def test_use_llm_false_skips_the_client(self):
        client = self.FakeClient({'characters': [{'name': 'Игнор'}]})
        records = parse_key_characters('Хорус — примарх', client=client, use_llm=False)
        self.assertEqual([r['name'] for r in records], ['Хорус'])
        self.assertEqual(client.calls, [])


class MergeTests(unittest.TestCase):
    def test_enriches_existing_and_appends_missing(self):
        characters = [
            {'normalized_name': 'Narrator', 'inferred_gender': 'unknown'},
            {'normalized_name': 'Хорус', 'inferred_gender': 'male'},
        ]
        key = parse_key_characters_heuristic(
            'Хорус Луперкаль — Воитель легиона\nСеватар — первый капитан'
        )
        merged = merge_key_characters(characters, key)
        names = [char['normalized_name'] for char in merged]
        self.assertEqual(names, ['Narrator', 'Хорус', 'Севатар'])

        horus = merged[1]
        self.assertTrue(horus['key_character'])
        self.assertEqual(horus['description'], 'Воитель легиона')
        self.assertEqual(horus['inferred_gender'], 'male')

        sevatar = merged[2]
        self.assertTrue(sevatar['key_character'])
        self.assertEqual(sevatar['inferred_age_category'], 'unknown')

    def test_key_entry_does_not_override_known_gender(self):
        characters = [{'normalized_name': 'Эуфратия', 'inferred_gender': 'female'}]
        key = [{'name': 'Эуфратия', 'aliases': [], 'gender': 'male',
                'age': 'elder', 'description': 'летописец'}]
        merged = merge_key_characters(characters, key)
        self.assertEqual(merged[0]['inferred_gender'], 'female')
        self.assertEqual(merged[0]['inferred_age_category'], 'elder')

    def test_two_similar_names_are_not_collapsed(self):
        characters = [
            {'normalized_name': 'ХорусЛуперсий'},
            {'normalized_name': 'ХорусАксиманд'},
        ]
        key = [
            {'name': 'Хорус Луперсий', 'aliases': [], 'gender': 'male',
             'age': 'adult', 'description': 'примарх'},
            {'name': 'Хорус Аксиманд', 'aliases': [], 'gender': 'male',
             'age': 'adult', 'description': 'капитан 5-й роты'},
        ]
        merged = merge_key_characters(characters, key)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]['description'], 'примарх')
        self.assertEqual(merged[1]['description'], 'капитан 5-й роты')

    def test_no_key_characters_returns_a_copy(self):
        characters = [{'normalized_name': 'Narrator'}]
        merged = merge_key_characters(characters, [])
        self.assertEqual(merged, characters)
        self.assertIsNot(merged, characters)

    def test_match_score_word_boundaries(self):
        self.assertEqual(_match_score('Хорус', 'Хорус'), 3)
        self.assertEqual(_match_score('Хорус', 'ХорусЛуперсий'), 1)
        self.assertEqual(_match_score('Петров', 'ИванПетров'), 1)
        # "Иван" is only a string prefix of "Иванов", not a word prefix.
        self.assertEqual(_match_score('Иван', 'Иванов'), 0)


if __name__ == '__main__':
    unittest.main()
