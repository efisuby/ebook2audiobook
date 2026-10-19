"""NLP utilities for Russian text processing."""

from __future__ import annotations

import re
from typing import Any, Optional

# spaCy publishes a single Russian news model. It is used for tokenization,
# sentence segmentation, POS tagging and PER entity recognition.
RU_SPACY_MODEL = "ru_core_news_sm"

# Every ``RussianBookNLP.process`` call needs the pipeline, and loading it from
# disk costs a second or so. It is stateless, so one instance is shared for the
# whole process.
_SPACY_MODEL_CACHE: dict = {}


def load_russian_spacy_model() -> Optional[Any]:
    """Load ``ru_core_news_sm``, returning ``None`` when it is not installed.

    The pipeline is cached, so repeated analysis runs in one process (for
    example the GUI) do not pay the load cost again.

    Install it with:
        uv pip install "$(python -m spacy info ru_core_news_sm --url)"
    """
    if RU_SPACY_MODEL in _SPACY_MODEL_CACHE:
        return _SPACY_MODEL_CACHE[RU_SPACY_MODEL]

    try:
        import spacy
    except ImportError:
        return None

    try:
        nlp = spacy.load(RU_SPACY_MODEL)
    except OSError:
        return None

    _SPACY_MODEL_CACHE[RU_SPACY_MODEL] = nlp
    return nlp


def get_ru_model_or_none(
    model_size: str = "small", model_path: Optional[str] = None
) -> Optional[Any]:
    """Backwards-compatible wrapper around :func:`load_russian_spacy_model`.

    ``model_size`` and ``model_path`` are accepted for API parity with the
    English BookNLP pipeline; only one Russian spaCy model exists, so both
    sizes resolve to ``ru_core_news_sm``.
    """
    return load_russian_spacy_model()


def normalize_cyrillic(text: str) -> str:
    """Normalize Russian text by replacing special characters.

    Replaces characters that look like standard Cyrillic letters but use
    non-standard code points.

    Args:
        text: Input string potentially containing non-standard characters.

    Returns:
        Normalized string with standard Cyrillic letters.
    """
    replacements = {
        "Ґ": "ґ",
        "Ѝ": "й",
        "Ѣ": "е",
        "ѳ": "ф",
        "ѣ": "е",
    }

    result = text
    for non_std, std in replacements.items():
        result = result.replace(non_std, std)
    return result


def normalize_text(text: str) -> str:
    """Normalize Russian text (alias for normalize_cyrillic)."""
    return normalize_cyrillic(text)


def detect_russian_punctuation(text: str) -> bool:
    """Detect whether text uses Russian punctuation conventions.

    Args:
        text: Input string to check.

    Returns:
        True if text contains typical Russian sentence punctuation.
    """
    russian_punct = {".", "!", "?", ":", ";", "…"}
    return any(p in text for p in russian_punct)


# Broad speech/attribution verbs used around quotes (speaker hints like
# «…», — подумала Анна). Matched by prefix so past-tense and aspect
# variants (сказал/сказала/скажет) are covered.
SPEECH_VERB_STEMS = (
    "сказ", "спрос", "ответ", "произнес", "произнёс", "воскликн",
    "промолч", "молч", "шепт", "прошепт", "заговор", "говор",
    "крикн", "проворч", "пробормот", "буркн", "нача", "продолж",
    "добав", "замет", "рассмея", "отрез", "усмехн", "улыбн", "вздохн",
    "окрикн", "переспрос", "возраз", "обрат", "оберн", "нарек",
    "озвуч", "переглян", "пробубни", "проблея", "пробурч", "процарап",
    "проискри", "вскрикн", "вопрос", "воспроизн", "отвеч", "дум",
    "подум", "произнош", "промелькн", "вымолв", "простон",
    "призна", "созна", "настаива", "подчерк", "упомян",
    "ворч", "бурч", "бормот",
)

# Report verbs only: these decide whether a dash segment is stage direction
# («— сказал Иван.», «— Пойдём, сказал он.») instead of speech. Deliberately
# narrower than SPEECH_VERB_STEMS — verbs like «начать» or «думать» occur
# *inside* utterances and would truncate them
# («— Туда, где можно начать сначала.» would end at «можно»).
ATTRIBUTION_VERB_STEMS = (
    "сказ", "говор", "произнес", "произнёс", "произнош",
    "спрос", "ответ", "отвеч", "вопрос", "переспрос",
    "воскликн", "вскрикн", "окрикн", "крикн",
    "промолч", "молч", "шепт", "прошепт", "заговор",
    "замет", "добав", "продолж", "отрез", "возраз", "настаива",
    "вымолв", "пробормот", "бормот", "проворч", "пробурч", "буркн",
    "ворч", "бурч", "нарек", "озвуч", "обрат", "подчерк", "упомян",
)


def _match_stem(word: str, stems: tuple) -> bool:
    lowered = word.strip().strip(".,!?;:…—–-«»\"'()").lower()
    if not lowered:
        return False
    return any(lowered.startswith(stem) for stem in stems)


def is_speech_verb(word: str) -> bool:
    """Return True when *word* looks like a Russian speech verb (broad set)."""
    return _match_stem(word, SPEECH_VERB_STEMS)


def is_attribution_verb(word: str) -> bool:
    """Return True for report verbs that turn a dash segment into stage direction."""
    return _match_stem(word, ATTRIBUTION_VERB_STEMS)


def strip_wrapping_punctuation(word: str) -> str:
    """Remove surrounding punctuation/quotes from a token."""
    return word.strip(".,!?;:…—–-«»\"'()").lower()
