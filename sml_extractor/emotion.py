"""Speaker emotion labels for the SML output.

The SML writer wraps every spoken line in ``[voice:...]``.  This module decides
what emotion the speaker is in, so the line can be wrapped in
``[emotion:LABEL]`` as well; the matching E2A patch (``lib/conf_models.py`` and
``lib/classes/tts_engines``) turns that label into inference parameters and a
volume tweak.

Labelling is done by the local in-process LLM, a batch of lines per request.  A
keyword and punctuation heuristic runs when the model is unavailable and patches
single lines the model skipped, so the SML always gets a label.  ``neutral`` is
the default and is never written out: a plain line is the shortest, and the
engine's own settings already read as neutral.
"""

from __future__ import annotations

import os
import re

from sml_extractor.llm_client import LlmError, get_client

# The vocabulary is deliberately small: every label must be meaningful to the
# E2A patch, and too fine a set makes the model inconsistent between calls.
DEFAULT_EMOTION = "neutral"
EMOTION_LABELS = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "afraid",
    "excited",
    "mysterious",
    "sarcastic",
    "melancholic",
    "energetic",
    "calm",
    "anxious",
    "passionate",
    "playful",
)
# Russian names and common synonyms, so the label survives translation or a
# model that decides to answer in the book's language.
_SYNONYMS = {
    "neutral": "neutral",
    "нейтральный": "neutral",
    "нейтрально": "neutral",
    "happy": "happy",
    "joy": "happy",
    "joyful": "happy",
    "радость": "happy",
    "радостный": "happy",
    "весёлый": "happy",
    "веселый": "happy",
    "sad": "sad",
    "грусть": "sad",
    "грустный": "sad",
    "печаль": "sad",
    "angry": "angry",
    "anger": "angry",
    "гнев": "angry",
    "злость": "angry",
    "злой": "angry",
    "afraid": "afraid",
    "fear": "afraid",
    "страх": "afraid",
    "испуг": "afraid",
    "excited": "excited",
    "восторг": "excited",
    "возбуждение": "excited",
    "mysterious": "mysterious",
    "загадочный": "mysterious",
    "таинственный": "mysterious",
    "sarcastic": "sarcastic",
    "сарказм": "sarcastic",
    "саркастичный": "sarcastic",
    "ироничный": "sarcastic",
    "melancholic": "melancholic",
    "меланхолия": "melancholic",
    "меланхоличный": "melancholic",
    "energetic": "energetic",
    "энергичный": "energetic",
    "бодрый": "energetic",
    "calm": "calm",
    "спокойный": "calm",
    "спокойствие": "calm",
    "anxious": "anxious",
    "тревога": "anxious",
    "тревожный": "anxious",
    "passionate": "passionate",
    "страстный": "passionate",
    "увлечённый": "passionate",
    "playful": "playful",
    "игривый": "playful",
    "шутливый": "playful",
}

# Keyword/typography fallback: (label, markers).  Order matters — the first
# match wins, strongest emotions first.
_HEURISTIC_RULES = (
    ("afraid", ("страшн", "боюс", "бойс", "пугает", "опасн", "дрожу", "ужас",
                "беги", "спасит", "afraid", "scared", "terror", "help me")),
    ("angry", ("злюсь", "ненавиж", "прекрати", "глупост", "достало", "молчать",
               "неправ", "вон", "убью", "проклят", "ярость", "гнев", "тварь",
               "angry", "furious", "shut up", "damn", "hate")),
    ("excited", ("невероятн", "восхит", "восторг", "не могу поверить", "вау",
                 "потрясающ", "ух ты", "amazing", "incredible", "wow")),
    ("anxious", ("тревож", "беспоко", "волну", "нервнич", "не знаю, что",
                 "worried", "nervous", "anxious")),
    ("sad", ("груст", "печаль", "плач", "слёз", "слез", "потер", "погиб",
             "прощай", "одинок", "больно", "горе", "скорб", "sad", "sorry",
             "grief", "mourn")),
    ("melancholic", ("тоск", "уныл", "меланхол", "задумчив", "вспомина",
                     "melancholy", "wistful")),
    ("happy", ("рад", "счаст", "весел", "улыб", "смех", "отлично", "отлично",
               "замечательн", "спасибо", "люблю", "прекрасно", "ура", "спасибо",
               "happy", "glad", "gladly", "joy", "great", "wonderful")),
    ("playful", ("шут", "шучу", "игрив", "хихи", "хихик", "игривая", "joke",
                 "kidding", "tease")),
    ("sarcastic", ("ну да", "конечно же", "как же", "разумеется", "очевидно же",
                   "sure,", "of course", "obviously", "yeah, right")),
    ("energetic", ("вперёд", "вперед", "быстрее", "давай", "бегом", "за мной",
                   "let's go", "hurry", "move")),
    ("passionate", ("клянусь", "люблю тебя", "всей душой", "навсегда",
                    "i love you", "forever")),
    ("calm", ("спокойн", "тихо", "не спеша", "всё хорошо", "все хорошо",
              "calm", "relax", "easy")),
    ("happy", ("ха-ха", "смех", "улыб")),
)

_SARCASM_TAIL_RE = re.compile(r"\.\.\.\s*$|…\s*$")
_QUOTED_ASIDE_RE = re.compile(r"\(\s*(смех|усмех|ирон|шутк|вздох|плач|крик|шёпот|шепот)")

# The cache keeps repeated short lines ("Да.", "Хорошо.") from being sent twice.
_CACHE: dict[str, str] = {}
_MAX_TEXT_CHARS = 400


def normalize_label(value) -> str:
    """Map anything the model returned onto a known label."""
    if not isinstance(value, str):
        return DEFAULT_EMOTION
    label = value.strip().lower().strip(".!:;,\"'")
    label = re.sub(r"[^\w\s-]", "", label, flags=re.UNICODE).strip()
    if label in _SYNONYMS:
        return _SYNONYMS[label]
    first = label.split()[0] if label else ""
    return _SYNONYMS.get(first, DEFAULT_EMOTION)


def heuristic_emotion(text: str) -> str:
    """Label a line from its words and punctuation alone."""
    if not text or not text.strip():
        return DEFAULT_EMOTION

    lowered = text.lower()
    if _QUOTED_ASIDE_RE.search(lowered):
        # "(с улыбкой)" and friends are authorial direction, not speech.
        return DEFAULT_EMOTION

    for label, markers in _HEURISTIC_RULES:
        if any(marker in lowered for marker in markers):
            return label

    if _SARCASM_TAIL_RE.search(text.strip()):
        return "melancholic"
    return DEFAULT_EMOTION


def clear_cache() -> None:
    """Forget cached labels (called before a new book)."""
    _CACHE.clear()


def _batch_size() -> int:
    try:
        return max(1, int(os.environ.get("E2A_SML_EMOTION_BATCH", "24")))
    except ValueError:
        return 24


def _system_prompt() -> str:
    labels = ", ".join(name for name in EMOTION_LABELS if name != DEFAULT_EMOTION)
    return (
        "You label the emotion a character is speaking with in a novel. "
        "You receive numbered lines of dialogue.\n"
        "Judge the spoken words together with their delivery cues (adverbs, "
        "stage directions such as \"in terror\" or \"холодно\") — the words "
        "alone can be misleading.\n"
        f"Answer with strict JSON only: {{\"emotions\": [...]}} — one label per "
        f"line, in the same order.\n"
        f"Allowed labels: {labels}, {DEFAULT_EMOTION}. Put a line that is plain, "
        "factual or unreadable in context under neutral. "
        "Use the punctuation, wording and delivery of the line; do not explain."
    )


def _user_prompt(texts: list[str]) -> str:
    lines = []
    for index, text in enumerate(texts, start=1):
        snippet = " ".join(text.split())[:_MAX_TEXT_CHARS]
        lines.append(f"{index}. {snippet}")
    return "\n".join(lines)


def _parse_batch(payload, size: int) -> list[str]:
    """Read a batch answer in any of the shapes the model may produce."""
    labels: list = []
    if isinstance(payload, dict):
        for key in ("emotions", "labels", "emotion", "results"):
            if key in payload:
                value = payload[key]
                labels = value if isinstance(value, list) else [value]
                break
        else:
            # {"1": "angry", ...} or {"1": {"emotion": "angry"}}
            keyed = []
            for index in range(1, size + 1):
                value = payload.get(str(index), payload.get(index))
                if isinstance(value, dict):
                    value = value.get("emotion", value.get("label"))
                keyed.append(value)
            labels = keyed
    elif isinstance(payload, list):
        labels = payload

    resolved = []
    for value in labels[:size]:
        if isinstance(value, dict):
            value = value.get("emotion", value.get("label"))
        resolved.append(normalize_label(value))
    return resolved


def analyze_emotions(
    texts: list[str],
    client=None,
    batch_size: int | None = None,
    progress=None,
) -> list[str]:
    """Label every text, in order.

    Args:
        texts: the spoken lines, in the order they appear in the book.
        client: optional LLM client (``get_client()`` by default).
        batch_size: lines per request.
        progress: optional ``progress(fraction, desc=...)`` callable.

    Returns:
        One label per input text.  Never raises: a broken model falls back to
        the heuristic for the affected batch.
    """
    if not texts:
        return []

    size = batch_size or _batch_size()
    client = client or get_client()
    # Repeated lines ("Да.", "Хорошо.") are labelled once per process.
    pending = [text for text in dict.fromkeys(texts) if text not in _CACHE]
    total = len(pending)

    for start in range(0, total, size):
        batch = pending[start:start + size]
        batch_labels: list = []
        try:
            payload = client.chat_json(
                _system_prompt(),
                _user_prompt(batch),
                temperature=0.0,
                max_tokens=len(batch) * 8 + 32,
            )
            batch_labels = _parse_batch(payload, len(batch))
        except LlmError as exc:
            print(f"[emotion] LLM labelling skipped for {len(batch)} lines: {exc}")
        except Exception as exc:  # a broken client must not kill the book
            print(f"[emotion] LLM labelling failed for {len(batch)} lines: {exc}")

        for offset, text in enumerate(batch):
            label = batch_labels[offset] if offset < len(batch_labels) else None
            _CACHE[text] = label or heuristic_emotion(text)

        if progress is not None:
            done = min(start + size, total)
            progress(done / total, desc=f"Emotion {done}/{total}")

    return [_CACHE[text] for text in texts]


def analyze_book_emotions(
    segments: list,
    client=None,
    progress=None,
) -> dict:
    """Label ``(speaker, text)`` segments and return a ``text -> label`` map.

    Narration (the ``Narrator`` speaker) keeps no emotion: it is read by the
    narrator voice and a labelled narration line adds noise, not expression.
    """
    dialogue = [
        text for speaker, text in segments
        if speaker and speaker != "Narrator" and text
    ]
    unique = list(dict.fromkeys(dialogue))
    labels = analyze_emotions(unique, client=client, progress=progress)
    return {
        text: label for text, label in zip(unique, labels)
        if label != DEFAULT_EMOTION
    }
