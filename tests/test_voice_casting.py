"""Tests for voice exclusion, the LLM client and LLM-assisted casting."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sml_extractor import voice_llm
from sml_extractor.llm_client import (
    LlmError,
    LlmClient,
    extract_json,
    model_candidates,
)
from sml_extractor.voice_matcher import (
    EXCLUDED_VOICE_NAMES,
    _find_best_voice,
    auto_assign_voices,
    is_voice_excluded,
    scan_voice_library,
)


class ExcludedVoiceTests(unittest.TestCase):
    def test_cartoon_and_meme_voices_are_excluded(self):
        for name in (
            'Cheburashka.mp3',
            'SSSR_Vinipuh.mp3',
            'SpoungeBob.mp3',
            'Zhirinovsky.mp3',
            'Deadpool.mp3',
            'YandexAlisa.mp3',
            'voices/rus/adult/male/Zadornov(koncert).mp3',
        ):
            self.assertTrue(is_voice_excluded(name), name)

    def test_regular_voices_are_kept(self):
        for name in (
            'AbdulovV.mp3',
            'ALEKSANDRA_KOSTINSKAYa.mp3',
            'Buhmin_AudioBook.mp3',
            'USSR_DOCS.mp3',
        ):
            self.assertFalse(is_voice_excluded(name), name)

    def test_environment_extends_the_list(self):
        self.assertFalse(is_voice_excluded('SomeCustomVoice.mp3'))
        with patch.dict(os.environ, {'E2A_SML_EXCLUDED_VOICES': 'SomeCustomVoice'}):
            self.assertTrue(is_voice_excluded('SomeCustomVoice.mp3'))

    def test_scan_voice_library_filters_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            male = Path(folder) / 'voices' / 'rus' / 'adult' / 'male'
            male.mkdir(parents=True)
            (male / 'AbdulovV.mp3').write_bytes(b'x')
            (male / 'Zhirinovsky.mp3').write_bytes(b'x')

            library = scan_voice_library(folder, 'rus')
        self.assertEqual(
            library['adult']['male'], ['voices/rus/adult/male/AbdulovV.mp3']
        )

    def test_best_voice_never_picks_an_excluded_voice(self):
        library = {'adult': {'male': ['Zhirinovsky.mp3', 'Cheburashka.mp3']}}
        self.assertIsNone(_find_best_voice('male', 'adult', library, set()))


class ExtractJsonTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(extract_json('{"a": 1}'), {'a': 1})

    def test_fenced_json(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {'a': 1})

    def test_json_inside_prose_and_reasoning(self):
        text = 'Thinking...\nThe answer is {"a": [1, 2]} and that is all.'
        self.assertEqual(extract_json(text), {'a': [1, 2]})

    def test_empty_answer_raises(self):
        with self.assertRaises(LlmError):
            extract_json('   ')

    def test_garbage_raises(self):
        with self.assertRaises(LlmError):
            extract_json('no json here')


class _FakeTensor:
    shape = (1, 3)

    def to(self, *args, **kwargs):
        return self


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, prompt, return_tensors=None):
        return {'input_ids': _FakeTensor()}

    def apply_chat_template(self, messages, **kwargs):
        if 'enable_thinking' in kwargs:
            raise ValueError('this template has no thinking switch')
        return 'rendered prompt'

    def decode(self, tokens, skip_special_tokens=False):
        return '{"ok": true}'


class _FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        return [[0, 1, 2, 3]]


class LocalLlmClientTests(unittest.TestCase):
    def test_available_and_chat_json_with_an_injected_generator(self):
        client = LlmClient(generator=lambda system, user, budget: '{"ok": true}')
        self.assertTrue(client.available())
        self.assertEqual(client.chat_json('s', 'u'), {'ok': True})

    def test_available_is_false_when_switched_off(self):
        with patch.dict(os.environ, {'E2A_SML_LLM_OFF': '1'}):
            self.assertFalse(LlmClient().available())

    def test_available_is_false_without_downloads_and_an_uncached_model(self):
        env = {'E2A_SML_LLM_ALLOW_DOWNLOAD': '0',
               'E2A_SML_LLM_MODEL': 'definitely/not-a-cached-model'}
        with patch.dict(os.environ, env):
            self.assertFalse(LlmClient().available())

    def test_candidates_come_from_the_environment(self):
        with patch.dict(os.environ, {'E2A_SML_LLM_MODEL': 'a/b, c/d'}):
            self.assertEqual(model_candidates(), ['a/b', 'c/d'])

    def test_a_model_that_fails_to_load_is_replaced_by_the_next_candidate(self):
        client = LlmClient(model='a/b, c/d')
        loader = unittest.mock.Mock(
            side_effect=[OSError('failed to load model'), (_FakeTokenizer(), _FakeModel())]
        )
        with patch('sml_extractor.llm_client._load_model', loader):
            self.assertEqual(client.chat_json('s', 'u'), {'ok': True})
        self.assertEqual([call.args[0] for call in loader.call_args_list],
                         ['a/b', 'c/d'])

    def test_the_answer_uses_the_chat_template_without_thinking(self):
        seen = {}

        def generator(system, user, budget):
            seen['budget'] = budget
            return '```json\n{"ok": 2}\n```'

        client = LlmClient(generator=generator)
        self.assertEqual(client.chat_json('s', 'u', max_tokens=64), {'ok': 2})
        self.assertEqual(seen['budget'], 64)

    def test_an_answer_without_json_raises(self):
        client = LlmClient(generator=lambda system, user, budget: 'no json here')
        with self.assertRaises(LlmError):
            client.chat_json('s', 'u')

    def test_unload_forgets_the_loaded_model(self):
        client = LlmClient(generator=lambda system, user, budget: '{}')
        client.unload()
        self.assertFalse(client.loaded)

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


LIBRARY = {
    'adult': {
        'female': [
            'voices/rus/adult/female/ALEKSANDRA_KOSTINSKAYa.mp3',
            'voices/rus/adult/female/Cheburashka.mp3',
            'voices/rus/adult/female/Kropina_YouTube.mp3',
        ],
        'male': [
            'voices/rus/adult/male/AbdulovV.mp3',
            'voices/rus/adult/male/Zhirinovsky.mp3',
            'voices/rus/adult/male/Buhmin_AudioBook.mp3',
        ],
    }
}


def make_characters():
    return [
        {'normalized_name': 'Narrator', 'inferred_gender': 'unknown',
         'inferred_age_category': 'adult'},
        {'normalized_name': 'Хорус', 'inferred_gender': 'male',
         'inferred_age_category': 'unknown', 'description': 'примарх'},
        {'normalized_name': 'Эуфратия', 'inferred_gender': 'unknown',
         'inferred_age_category': 'unknown', 'description': 'молодая женщина'},
    ]


class VoiceCastingTests(unittest.TestCase):
    def test_candidates_skip_excluded_library_voices(self):
        names = [c['name'] for c in voice_llm._candidates(LIBRARY)]
        self.assertNotIn('Cheburashka', names)
        self.assertNotIn('Zhirinovsky', names)
        self.assertIn('AbdulovV', names)

    def test_custom_voices_are_never_excluded(self):
        names = [c['name'] for c in
                 voice_llm._candidates({}, ['/tmp/Cheburashka.wav'])]
        self.assertEqual(names, ['Cheburashka'])

    def test_llm_choice_is_applied_and_reasons_kept(self):
        characters = make_characters()
        client = FakeClient({'assignments': [
            {'character': 'Хорус', 'voice': 'AbdulovV', 'why': 'низкий голос'},
            {'character': 'Эуфратия', 'voice': 'ALEKSANDRA_KOSTINSKAYa',
             'gender': 'female', 'age': 'adult', 'why': 'молодой женский'},
        ]})
        assignments, reasons = voice_llm.assign_voices_with_llm(
            characters, LIBRARY, client=client, language='rus'
        )
        self.assertEqual(assignments['Хорус'], 'voices/rus/adult/male/AbdulovV.mp3')
        self.assertEqual(
            assignments['Эуфратия'], 'voices/rus/adult/female/ALEKSANDRA_KOSTINSKAYa.mp3'
        )
        self.assertEqual(reasons['Хорус'], 'низкий голос')
        # Unknown gender/age are filled from the LLM answer.
        eufratia = next(c for c in characters if c['normalized_name'] == 'Эуфратия')
        self.assertEqual(eufratia['inferred_gender'], 'female')
        self.assertEqual(eufratia['inferred_age_category'], 'adult')

    def test_wrong_gender_and_unknown_voice_are_rejected(self):
        characters = make_characters()
        client = FakeClient({'assignments': [
            {'character': 'Хорус', 'voice': 'ALEKSANDRA_KOSTINSKAYa', 'why': 'нет'},
            {'character': 'Эуфратия', 'voice': 'NoSuchVoice', 'why': 'нет'},
        ]})
        assignments, reasons = voice_llm.assign_voices_with_llm(
            characters, LIBRARY, client=client, language='rus'
        )
        # Both fall back to a matching voice; reasons never explain a fallback.
        self.assertTrue(assignments['Хорус'].endswith('male/AbdulovV.mp3')
                        or assignments['Хорус'].endswith('male/Buhmin_AudioBook.mp3'))
        self.assertTrue(assignments['Эуфратия'].endswith('female/Kropina_YouTube.mp3')
                        or assignments['Эуфратия'].endswith('female/ALEKSANDRA_KOSTINSKAYa.mp3'))
        self.assertNotIn('Хорус', reasons)
        self.assertNotIn('Эуфратия', reasons)

    def test_voices_stay_distinct(self):
        characters = make_characters()
        client = FakeClient({'assignments': [
            {'character': 'Хорус', 'voice': 'AbdulovV'},
            {'character': 'Эуфратия', 'voice': 'AbdulovV'},
        ]})
        assignments, _ = voice_llm.assign_voices_with_llm(
            characters, LIBRARY, client=client, language='rus'
        )
        self.assertEqual(len(set(assignments.values())), len(assignments))

    def test_llm_error_returns_nothing(self):
        assignments, reasons = voice_llm.assign_voices_with_llm(
            make_characters(), LIBRARY,
            client=FakeClient(error=LlmError('offline')), language='rus',
        )
        self.assertEqual(assignments, {})
        self.assertEqual(reasons, {})

    def test_prompt_lists_characters_and_voices(self):
        client = FakeClient({'assignments': []})
        voice_llm.assign_voices_with_llm(
            make_characters(), LIBRARY, client=client, language='rus'
        )
        system, user = client.calls[0]
        self.assertIn('Russian', system)
        self.assertIn('Хорус', user)
        self.assertIn('AbdulovV', user)
        self.assertNotIn('Zhirinovsky', user)


class SelectCastTests(unittest.TestCase):
    def make_crowd(self):
        characters = [
            {'normalized_name': f'Speak{i}', 'inferred_gender': 'male',
             'inferred_age_category': 'adult'}
            for i in range(10)
        ]
        characters.append({'normalized_name': 'Гарвель Локен', 'key_character': True,
                           'inferred_gender': 'male', 'inferred_age_category': 'adult'})
        characters.append({'normalized_name': 'Narrator', 'inferred_gender': 'unknown',
                           'inferred_age_category': 'adult'})
        return characters

    def test_everything_is_kept_below_the_limit(self):
        characters = self.make_crowd()
        self.assertEqual(voice_llm.select_cast(characters, limit=100), characters)

    def test_key_characters_and_narrator_are_always_included(self):
        characters = self.make_crowd()
        select = voice_llm.select_cast(characters, {'Speak0': 900}, limit=3)
        names = [c['normalized_name'] for c in select]
        self.assertIn('Гарвель Локен', names)
        self.assertIn('Narrator', names)
        self.assertEqual(len(select), 3)

    def test_busiest_characters_fill_the_remaining_slots(self):
        characters = self.make_crowd()
        select = voice_llm.select_cast(
            characters, {'Speak7': 100, 'Speak3': 50}, limit=4
        )
        names = [c['normalized_name'] for c in select]
        self.assertIn('Speak7', names)
        self.assertIn('Speak3', names)

    def test_the_prompt_is_bounded_for_a_large_book(self):
        characters = self.make_crowd() * 20  # 240 speakers
        client = FakeClient({'assignments': []})
        voice_llm.assign_voices_with_llm(
            characters, LIBRARY, client=client, language='rus',
            counts={'Speak9': 10},
        )
        _system, user = client.calls[0]
        # 40 characters + the voice list, never the full crowd.
        self.assertLessEqual(user.count('\n- '), voice_llm.MAX_CAST + 40)
        self.assertNotIn('Speak8', user)

    def test_a_bounded_cast_still_fills_everyone(self):
        characters = self.make_crowd() * 20
        client = FakeClient({'assignments': [
            {'character': 'Narrator', 'voice': 'ALEKSANDRA_KOSTINSKAYa'},
            {'character': 'Гарвель Локен', 'voice': 'AbdulovV'},
        ]})
        assignments, reasons = voice_llm.assign_voices_with_llm(
            characters, LIBRARY, client=client, language='rus', counts={},
        )
        # The result covers the whole cast, not only the characters the LLM saw.
        self.assertEqual(
            assignments['Narrator'],
            'voices/rus/adult/female/ALEKSANDRA_KOSTINSKAYa.mp3',
        )
        self.assertEqual(
            assignments['Гарвель Локен'], 'voices/rus/adult/male/AbdulovV.mp3'
        )
        # Every distinct name is cast (the repeated crowd holds 12 of them),
        # including the ones the LLM never saw; the four-voice library is
        # smaller than the cast, so the surplus has to share voices.
        self.assertEqual(len(assignments), 12)
        self.assertIn('Speak9', assignments)
        self.assertLess(len(set(assignments.values())), 12)

    def test_key_characters_get_a_voice_before_the_long_tail(self):
        # Three voices, and the key character is buried in the crowd.
        library = {
            'adult': {
                'male': ['voices/rus/adult/male/One.mp3',
                         'voices/rus/adult/male/Two.mp3'],
                'female': ['voices/rus/adult/female/Three.mp3'],
            },
            'child': {'male': [], 'female': []},
            'teen': {'male': [], 'female': []},
            'elder': {'male': [], 'female': []},
        }
        characters = self.make_crowd() * 20
        client = FakeClient({'assignments': []})
        assignments, _ = voice_llm.assign_voices_with_llm(
            characters, library, client=client, language='rus', counts={},
        )
        # The key character is cast first, so it gets one of the two male
        # voices; in plain list order the ten Speak* males would take them.
        self.assertTrue(
            assignments['Гарвель Локен'].endswith(('male/One.mp3', 'male/Two.mp3'))
        )
        self.assertEqual(len(assignments), 12)
        self.assertLessEqual(len(set(assignments.values())), 3)

    def test_quote_counts_maps_coref_ids_to_names(self):
        booknlp_data = {
            'book_data': {'characters': [
                {'id': 1, 'mentions': {'proper': [{'n': 'Хорус'}]}},
                {'id': 2, 'mentions': {'proper': [{'n': 'Локен'}]}},
            ]},
            'quotes': [
                {'char_id': 1}, {'char_id': 1}, {'char_id': 2}, {'char_id': None},
            ],
        }
        counts = voice_llm.quote_counts(booknlp_data)
        self.assertEqual(counts.get('Хорус'), 2)
        self.assertEqual(counts.get('Локен'), 1)
        self.assertEqual(counts.get('Narrator'), 1)
        self.assertEqual(voice_llm.quote_counts({}), {})


class MatcherReuseTests(unittest.TestCase):
    """The plain matcher is used when no LLM is available."""

    @staticmethod
    def _library():
        return {
            'adult': {
                'male': ['voices/rus/adult/male/One.mp3',
                         'voices/rus/adult/male/Two.mp3'],
                'female': ['voices/rus/adult/female/Three.mp3'],
            },
            'child': {'male': [], 'female': []},
            'teen': {'male': [], 'female': []},
            'elder': {'male': [], 'female': []},
        }

    @staticmethod
    def _crowd(count):
        return [
            {'normalized_name': f'C{i}', 'inferred_gender': 'male',
             'inferred_age_category': 'adult'}
            for i in range(count)
        ]

    def test_reuse_fills_the_cast_when_the_pool_runs_out(self):
        assignments = auto_assign_voices(self._crowd(6), self._library())
        self.assertEqual(len(assignments), 6)
        self.assertLessEqual(len(set(assignments.values())), 3)

    def test_reuse_can_be_disabled(self):
        assignments = auto_assign_voices(
            self._crowd(6), self._library(), allow_reuse=False
        )
        self.assertLessEqual(len(assignments), 3)

    def test_key_characters_are_cast_first(self):
        characters = self._crowd(6) + [
            {'normalized_name': 'Гарвель Локен', 'inferred_gender': 'male',
             'inferred_age_category': 'adult', 'key_character': True},
        ]
        assignments = auto_assign_voices(
            characters, self._library(), allow_reuse=False
        )
        self.assertTrue(
            assignments['Гарвель Локен'].endswith(('One.mp3', 'Two.mp3'))
        )


if __name__ == '__main__':
    unittest.main()
