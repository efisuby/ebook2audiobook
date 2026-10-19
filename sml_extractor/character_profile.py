"""Short per-character descriptions.

The casting screen needs to say who a character is before a voice is picked,
and BookNLP only provides a name, a gender and an age category.  This module
reads the book itself: it collects each character's own lines and the narration
around the mentions of their name, then asks the local instruct model
(:mod:`sml_extractor.llm_client`) for a one-sentence description.

The heuristic below is always computed first, so a missing or broken model
still leaves a usable description instead of an empty field, and a description
the user wrote by hand (through the key-character notes) is never overwritten.
"""

from __future__ import annotations

import os
import re

from sml_extractor.llm_client import LlmError, get_client
from sml_extractor.sml_generator import (
    _book_segments,
    _normalize_name,
    clean_tts_text,
)

NARRATOR = "Narrator"

# How much narration around a name mention is kept as evidence.
_WINDOW = 200
# Longest snippet handed to the model / kept for one quote.
_SNIPPET = 220
# Hard cap on a description.
_MAX_DESCRIPTION = 300
# Without a single line of their own a character needs at least this much
# narration to be worth a model call: below it the heuristic is safer than a
# model that has almost nothing to read.
_MIN_MENTIONS = 2

_SOURCE_USER = "user"
_SOURCE_LLM = "llm"
_SOURCE_HEURISTIC = "heuristic"

_AGE_PHRASES = {
    "rus": {
        ("male", "child"): "мальчик",
        ("male", "teen"): "юноша",
        ("male", "adult"): "взрослый мужчина",
        ("male", "elder"): "пожилой мужчина",
        ("female", "child"): "девочка",
        ("female", "teen"): "девушка",
        ("female", "adult"): "взрослая женщина",
        ("female", "elder"): "пожилая женщина",
        ("unknown", "child"): "ребёнок",
        ("unknown", "teen"): "подросток",
        ("unknown", "adult"): "персонаж",
        ("unknown", "elder"): "пожилой персонаж",
    },
    "eng": {
        ("male", "child"): "a boy",
        ("male", "teen"): "a teenage boy",
        ("male", "adult"): "an adult man",
        ("male", "elder"): "an elderly man",
        ("female", "child"): "a girl",
        ("female", "teen"): "a teenage girl",
        ("female", "adult"): "an adult woman",
        ("female", "elder"): "an elderly woman",
        ("unknown", "child"): "a child",
        ("unknown", "teen"): "a teenager",
        ("unknown", "adult"): "a character",
        ("unknown", "elder"): "an elderly character",
    },
}

_NARRATOR_PHRASES = {
    "rus": "Голос рассказчика, ведёт повествование книги.",
    "eng": "The narrator, reading the story.",
}

_LINE_LABELS = {
    "rus": {"lines": "реплик", "words": "Характерные слова"},
    "eng": {"lines": "lines", "words": "Common words"},
}

# Words that say nothing about a character.  Deliberately short: a real
# frequency list would also drop rarer but more telling words.
_STOPWORDS = {
    # Russian
    "это", "этот", "эта", "эти", "того", "тому", "этом", "этой", "чтобы", "когда",
    "если", "уже", "ещё", "еще", "или", "быть", "был", "была", "были", "было",
    "есть", "него", "неё", "нее", "нему", "ней", "них", "себе", "себя", "свой",
    "свою", "свои", "весь", "вся", "все", "всё", "всех", "всю", "очень", "тоже",
    "также", "только", "меня", "тебя", "вас", "нас", "мне", "тебе", "вам", "нам",
    "его", "её", "ее", "ему", "ей", "им", "ими", "для", "без", "при", "про",
    "над", "под", "через", "между", "потом", "тогда", "здесь", "там", "тут",
    "сейчас", "теперь", "всегда", "никогда", "ничего", "нибудь", "кто", "что",
    "чего", "чем", "кому", "кого", "как", "какой", "какая", "какие", "такой",
    "какая", "такие", "как-то", "что-то", "кто-то", "чём", "чем", "какие-то",
    "себя", "ведь", "хоть", "пусть", "даже", "разве", "только", "почти", "вовсе",
    "очень", "слишком", "совсем", "сразу", "потом", "затем", "опять", "снова",
    "здесь", "куда", "откуда", "почему", "зачем", "сколько", "надо", "нужно",
    "можно", "нельзя", "должен", "должна", "должны", "хочу", "хочешь", "хочет",
    "хотим", "хотят", "знаю", "знаешь", "знает", "знаем", "знают", "думаю",
    "думаешь", "думает", "говорю", "говоришь", "говорит", "говорят", "сказал",
    "сказала", "сказали", "говорил", "говорила", "ответил", "ответила", "спросил",
    "спросила", "молчал", "молчала", "смотрел", "смотрела", "видел", "видела",
    "понимаю", "понимаешь", "понимает", "вижу", "видишь", "видит", "помню",
    "помнишь", "помнит", "хотел", "хотела", "хотелось", "стало", "стал", "стала",
    "будет", "будут", "буду", "будешь", "будем", "если", "чтобы", "пока", "после",
    "перед", "просто", "прямо", "вообще", "конечно", "наверное", "кажется",
    "пожалуй", "впрочем", "однако", "кстати", "наконец", "например", "именно",
    "тоже", "также", "такое", "такая", "такой", "этакий", "нечто", "некий",
    "каждый", "любой", "другой", "другие", "прочие", "остальные", "самый",
    "самая", "самые", "меня", "меня", "тебе", "тебя", "себя", "себе",
    # English
    "the", "and", "that", "this", "these", "those", "there", "their", "them",
    "they", "with", "have", "has", "had", "been", "being", "will", "would",
    "should", "could", "must", "just", "very", "really", "only", "also", "than",
    "then", "when", "where", "what", "which", "while", "your", "yours", "about",
    "into", "over", "some", "such", "each", "other", "another", "because",
    "before", "after", "again", "still", "even", "ever", "never", "always",
    "know", "think", "thought", "said", "says", "tell", "told", "well", "yeah",
    "okay", "please", "thanks", "sorry", "thing", "things", "something",
    "anything", "nothing", "everything", "someone", "anyone", "everyone",
    "himself", "herself", "myself", "yourself", "would", "could", "should",
    "much", "many", "more", "most", "less", "least", "own", "same", "here",
}


def _language_key(language: str | None) -> str:
    return "rus" if str(language or "").lower().startswith(("ru", "rus")) else "eng"


def _batch_size() -> int:
    try:
        return max(1, int(os.environ.get("E2A_SML_PROFILE_BATCH", "8")))
    except ValueError:
        return 8


def _character_name(character: dict) -> str:
    for key in ("normalized_name", "name", "character"):
        value = character.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _split_camel(name: str) -> list[str]:
    """``ПётрИванов`` -> ``['Пётр', 'Иванов']`` (works for Cyrillic too)."""
    return [part for part in re.split(r"(?<=[a-zа-яё])(?=[A-ZА-ЯЁ])", name) if part]


def _clean_text(value) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()[:_MAX_DESCRIPTION]


def _match_key(value: str) -> str:
    """Key that ignores spacing and case: ``Пётр Иванов`` == ``ПётрИванов``."""
    return re.sub(r"[^\w]", "", str(value or ""), flags=re.UNICODE).casefold()


def _window(text: str, start: int, end: int, width: int = _WINDOW) -> str:
    """Narration around a match, trimmed back to word boundaries."""
    left = max(0, start - width)
    right = min(len(text), end + width)
    snippet = text[left:right]
    if left > 0:
        cut = snippet.find(" ")
        if 0 <= cut < 40:
            snippet = snippet[cut + 1:]
    if right < len(text):
        cut = snippet.rfind(" ")
        if cut > len(snippet) - 40:
            snippet = snippet[:cut]
    return re.sub(r"\s+", " ", snippet).strip()


def _sentences(text: str) -> list[str]:
    """Split narration into sentences.

    A mention is taken from the sentence the name occurs in (not from a fixed
    character window), so the material of one character never bleeds into the
    description of another one met in the next sentence.
    """
    return [item.strip() for item in re.split(r"(?<=[.!?…])\s+", text) if item.strip()]


def _mention_index(booknlp_data: dict | None) -> dict:
    """``{normalized name: [spellings BookNLP saw]}`` from the .book data."""
    book_data = (booknlp_data or {}).get("book_data") or {}
    index: dict[str, list[str]] = {}
    for character in book_data.get("characters", []) or []:
        mentions = character.get("mentions") or {}
        names = []
        for group in ("proper", "common"):
            for mention in mentions.get(group, []) or []:
                value = mention.get("n") if isinstance(mention, dict) else mention
                if isinstance(value, str) and value.strip():
                    names.append(re.sub(r"\s+", " ", value).strip())
        if not names:
            continue
        bucket = index.setdefault(_normalize_name(names[0]), [])
        for value in names:
            if value not in bucket:
                bucket.append(value)
    return index


def _variants_for(character: dict, mention_index: dict) -> list[str]:
    """Spellings worth searching for in the narration, longest ones first."""
    name = _character_name(character)
    if not name:
        return []

    variants: list[str] = []
    seen: set[str] = set()

    def add(value: str):
        value = re.sub(r"\s+", " ", value or "").strip(" .,;:!?—–-\"'«»")
        if len(value) < 3:
            return
        key = value.casefold()
        if key in seen:
            return
        seen.add(key)
        variants.append(value)

    for value in mention_index.get(name, []):
        add(value)

    words = _split_camel(name)
    if len(words) > 1:
        add(" ".join(words))
        add(max(words, key=len))
    add(name)
    return variants


def _dialogue_keys(speaker: str) -> set:
    """Keys a dialogue speaker answers to (``Мария`` also matches ``МарияПетрова``)."""
    keys = {_match_key(speaker), _normalize_name(speaker)}
    words = _split_camel(_normalize_name(speaker))
    if words:
        keys.add(_match_key(words[0]))
        if len(words) > 1:
            keys.add(_match_key(words[-1]))
    return {key for key in keys if key}


def _dialogue_index(booknlp_data: dict) -> dict:
    """``{key: {speaker: lines}}`` for every spoken line of the book."""
    index: dict[str, dict[str, list[str]]] = {}
    for speaker, text in _book_segments(booknlp_data):
        if speaker is None or speaker == NARRATOR:
            continue
        cleaned = clean_tts_text(text)
        if not cleaned:
            continue
        for key in _dialogue_keys(speaker):
            index.setdefault(key, {}).setdefault(speaker, []).append(cleaned)
    return index


def _bucket_lines(bucket: dict) -> list:
    """The lines of one character; the longest list wins on an ambiguous key."""
    lines: list = []
    for items in bucket.values():
        if len(items) > len(lines):
            lines = items
    return list(lines)


def _dialogue_for(name: str, index: dict) -> list:
    """Own lines of ``name`` — by full name first, then by a unique first name."""
    bucket = index.get(_match_key(name))
    if bucket:
        return _bucket_lines(bucket)

    words = _split_camel(name)
    if len(words) > 1:
        bucket = index.get(_match_key(words[0]))
        if bucket and len(bucket) == 1:
            return _bucket_lines(bucket)
    return []


def _narration(booknlp_data: dict) -> str:
    pieces = []
    for speaker, text in _book_segments(booknlp_data):
        if not text:
            continue
        if speaker is None or speaker == NARRATOR:
            pieces.append(text)
    return "\n".join(pieces)


def _pick_quotes(quotes: list[str], limit: int) -> list[str]:
    """Keep the longest ``limit`` lines, in their original order."""
    if limit <= 0:
        return []
    ranked = sorted(range(len(quotes)), key=lambda i: -len(quotes[i]))[:limit]
    return [quotes[i] for i in sorted(ranked)]


def build_evidence(
    booknlp_data: dict | None,
    characters: list | None = None,
    max_quotes: int = 4,
    max_mentions: int = 3,
    width: int = _WINDOW,
) -> dict:
    """Collect the book material a description can be written from.

    Returns ``{name: {'quotes', 'quote_count', 'mentions'}}`` — the character's
    own lines (longest first, capped) and narration snippets built around the
    occurrences of the name.  The trailing word boundary is intentionally
    omitted when searching so Russian case endings still match.
    """
    characters = list(characters or [])
    try:
        dialogue = _dialogue_index(booknlp_data or {})
        narration = _narration(booknlp_data or {})
    except (ValueError, KeyError, AttributeError):
        dialogue, narration = {}, ""

    mention_index = _mention_index(booknlp_data or {})

    if characters:
        names = [name for name in map(_character_name, characters) if name]
    else:
        names = []
        for name in list(mention_index) + list(dialogue):
            if name not in names:
                names.append(name)

    by_name = {_match_key(_character_name(item)): item for item in characters}
    evidence: dict[str, dict] = {}

    for name in names:
        character = by_name.get(_match_key(name), {"normalized_name": name})

        own_lines = _dialogue_for(name, dialogue)

        mentions: list[str] = []
        seen: set[str] = set()
        sentences = _sentences(narration)
        for variant in _variants_for(character, mention_index):
            if len(mentions) >= max_mentions:
                break
            pattern = re.compile(r"\b" + re.escape(variant), re.IGNORECASE | re.UNICODE)
            for sentence in sentences:
                if len(mentions) >= max_mentions:
                    break
                if not pattern.search(sentence):
                    continue
                snippet = sentence[: max(width, _SNIPPET)]
                key = snippet[:80].casefold()
                if key in seen:
                    continue
                seen.add(key)
                mentions.append(snippet)

        if name == NARRATOR and not mentions and narration:
            mentions = [
                sentence[: max(width, _SNIPPET)]
                for sentence in sentences
                if len(sentence) >= 20
            ][:max_mentions]

        evidence[name] = {
            "quotes": _pick_quotes(own_lines, max_quotes),
            "quote_count": len(own_lines),
            "mentions": mentions,
        }

    return evidence


def _words(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[^\W\d_]{4,}", str(text or "").lower(), flags=re.UNICODE)
    ]


def characteristic_words(
    evidence: dict,
    limit: int = 5,
    max_doc_ratio: float = 0.25,
) -> dict:
    """Words that set a character apart from the rest of the cast.

    Words shared by more than ``max_doc_ratio`` of the cast carry no signal and
    are dropped, as are stopwords, numbers and short words.
    """
    counts: dict[str, dict[str, int]] = {}
    for name, data in (evidence or {}).items():
        counter: dict[str, int] = {}
        for text in list(data.get("quotes", [])) + list(data.get("mentions", [])):
            for word in _words(text):
                if word in _STOPWORDS:
                    continue
                counter[word] = counter.get(word, 0) + 1
        counts[name] = counter

    threshold = max(2, int(len(evidence or {}) * max_doc_ratio))
    document_frequency: dict[str, int] = {}
    for name, counter in counts.items():
        for word in counter:
            document_frequency[word] = document_frequency.get(word, 0) + 1

    result: dict[str, list[str]] = {}
    for name, counter in counts.items():
        ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        result[name] = [
            word
            for word, _ in ranked
            if document_frequency.get(word, 0) < threshold
        ][: max(0, limit)]
    return result


def heuristic_description(
    character: dict,
    evidence: dict | None = None,
    language: str = "rus",
    words: list | None = None,
) -> str:
    """A description built only from the BookNLP attributes and the text."""
    lang = _language_key(language)
    name = _character_name(character)

    if name == NARRATOR:
        return _NARRATOR_PHRASES[lang]

    gender = str(
        character.get("inferred_gender") or character.get("gender") or "unknown"
    ).lower()
    age = str(
        character.get("inferred_age_category") or character.get("age") or "adult"
    ).lower()
    phrase = _AGE_PHRASES[lang].get((gender, age)) or _AGE_PHRASES[lang][
        ("unknown", "adult")
    ]

    labels = _LINE_LABELS[lang]
    quote_count = int((evidence or {}).get("quote_count") or 0)
    parts = [f"{phrase}, {quote_count} {labels['lines']}"]

    words = [word for word in (words or []) if word]
    if words:
        parts.append(f"{labels['words']}: {', '.join(words)}")

    description = ". ".join(parts) + "."
    return description[0].upper() + description[1:]


def _system_prompt(language: str) -> str:
    if _language_key(language) == "rus":
        return (
            "Ты готовишь описания персонажей книги для подбора голоса в студии "
            "озвучивания. По репликам персонажа и по фрагментам повествования "
            "вокруг него напиши ОДНО короткое описание (до 160 символов) на "
            "русском языке: кто это, каков он по характеру и как говорит.\n"
            "Правила:\n"
            "- опирайся только на приведённые фрагменты, биографию не додумывай;\n"
            "- пол и возраст из списка — факты, не противоречь им;\n"
            "- не называй возраст числом и не придумывай внешность, которой нет во фрагментах;\n"
            "- пиши только про ЭТОГО персонажа и не переноси на него черты других из списка;\n"
            "- если фрагментов мало, ограничься тем, что в них видно (манера речи, "
            "характер, роль в сцене).\n"
            "- пиши связным текстом, без заголовков и подписей вроде «Внешность:»; "
            "не начинай с фразы «Возрастная категория».\n"
            "Ответ верни строго в формате JSON:\n"
            '{"descriptions": [{"character": "<имя как в списке>", "description": "..."}]}\n'
            "Порядок записей — как в списке."
        )
    return (
        "You prepare character profiles that a casting director uses to pick a "
        "voice for an audiobook. From the character's own lines and the "
        "narration around them write ONE short description (up to 160 "
        "characters): who the character is, how they behave and how they "
        "speak.\n"
        "Rules:\n"
        "- use only the provided fragments, do not invent a biography;\n"
        "- the gender and age in the list are facts, stay consistent with them;\n"
        "- do not turn the age into a number and do not invent looks the fragments do not mention;\n"
        "- describe only THIS character, do not move traits of others in the list onto them;\n"
        "- if the fragments are few, stick to what they show (manner of speech, "
        "temper, role in the scene).\n"
        "- write flowing prose, without headings or labels such as \"Appearance:\".\n"
        "Answer strictly as JSON:\n"
        '{"descriptions": [{"character": "<name as in the list>", "description": "..."}]}\n'
        "Keep the given order."
    )


def _user_prompt(chunk: list, evidence: dict, language: str) -> str:
    russian = _language_key(language) == "rus"
    lines_label = "Реплики" if russian else "Lines"
    context_label = "Контекст" if russian else "Context"
    quote_open, quote_close = ("«", "»") if russian else ('"', '"')

    blocks = []
    for index, (_, character) in enumerate(chunk, start=1):
        name = _character_name(character)
        gender = character.get("inferred_gender") or character.get("gender") or "unknown"
        age = (
            character.get("inferred_age_category")
            or character.get("age")
            or "adult"
        )
        block = [f"{index}. {name} | {gender} | {age}"]
        data = evidence.get(name, {})

        quotes = [item[:_SNIPPET] for item in data.get("quotes", [])]
        if quotes:
            block.append(
                "   " + lines_label + ": "
                + " | ".join(f"{quote_open}{item}{quote_close}" for item in quotes)
            )
        mentions = [item[:_SNIPPET] for item in data.get("mentions", [])]
        if mentions:
            block.append(
                "   " + context_label + ": "
                + " | ".join(f"{quote_open}{item}{quote_close}" for item in mentions)
            )
        blocks.append("\n".join(block))
    return "\n".join(blocks)


def _lookup(by_name: dict, name: str):
    """Find a record by character name, tolerating spacing and short forms."""
    key = _match_key(name)
    if not key:
        return None
    if key in by_name:
        return by_name[key]
    for candidate, value in by_name.items():
        if candidate and (key.startswith(candidate) or candidate.startswith(key)):
            return value
    return None


def _value_text(value) -> str | None:
    if isinstance(value, dict):
        value = value.get("description") or value.get("profile") or value.get("text")
    if isinstance(value, list) and value:
        value = value[0]
    if not isinstance(value, str):
        return None
    return _clean_text(value) or None


def _parse_batch(payload, size: int, names: list | None = None) -> list:
    """Tolerantly read one batch of ``{"character", "description"}`` records."""
    names = list(names or [])
    descriptions: list = [None] * size

    records = payload
    if isinstance(payload, dict):
        for key in ("descriptions", "characters", "profiles", "results", "items", "data"):
            value = payload.get(key)
            if isinstance(value, (list, dict)):
                records = value
                break

    if isinstance(records, dict):
        by_name = {_match_key(key): value for key, value in records.items()}
        for index in range(size):
            value = _lookup(by_name, names[index]) if index < len(names) else None
            if value is None:
                value = records.get(str(index + 1), records.get(index + 1))
            if value is None:
                value = records.get(str(index), records.get(index))
            descriptions[index] = _value_text(value)
        return descriptions

    positional: list = []
    by_name: dict[str, object] = {}
    for item in records if isinstance(records, list) else []:
        if isinstance(item, dict):
            name = (
                item.get("character")
                or item.get("name")
                or item.get("character_name")
            )
            value = (
                item.get("description")
                or item.get("profile")
                or item.get("text")
                or item.get("desc")
            )
            if value is None and len(item) == 1:
                value = next(iter(item.values()))
            if name is not None:
                by_name[_match_key(name)] = value
                continue
            positional.append(value)
        else:
            positional.append(item)

    for index in range(size):
        value = _lookup(by_name, names[index]) if index < len(names) else None
        if value is None and index < len(positional):
            value = positional[index]
        descriptions[index] = _value_text(value)
    return descriptions


def describe_characters(
    characters: list,
    booknlp_data: dict | None = None,
    client=None,
    use_llm: bool = True,
    language: str = "rus",
    batch_size: int | None = None,
    progress=None,
    limit: int | None = None,
) -> list:
    """Give every character a ``description`` and a ``description_source``.

    The heuristic is written first; the model then replaces it for the
    characters it answers for.  A description already present without a source
    (or with the ``user`` source) is left untouched — that is the note the user
    typed in, not something to regenerate.  The narrator and characters with no
    lines of their own and almost no narration are not sent to the model at
    all: with so little to read it invents, while the heuristic stays factual
    (see ``_MIN_MENTIONS``).  Never raises.
    """
    characters = list(characters or [])
    evidence = build_evidence(booknlp_data, characters)
    words = characteristic_words(evidence)

    pending = []
    for index, character in enumerate(characters):
        name = _character_name(character)
        existing = _clean_text(character.get("description"))
        source = character.get("description_source")
        if existing and source in (None, _SOURCE_USER):
            character["description"] = existing
            character["description_source"] = _SOURCE_USER
            continue
        character["description"] = heuristic_description(
            character, evidence.get(name, {}), language=language, words=words.get(name)
        )
        character["description_source"] = _SOURCE_HEURISTIC
        data = evidence.get(name, {})
        if name == NARRATOR:
            # the narrator is not cast from the text, its phrase is fixed
            continue
        if data.get("quotes") or len(data.get("mentions") or []) >= _MIN_MENTIONS:
            pending.append((index, character))

    if not use_llm or not pending:
        return characters

    if limit is not None and limit > 0:
        pending = pending[: int(limit)]

    client = client or get_client()
    size = batch_size or _batch_size()
    total = len(pending)
    done = 0

    for start in range(0, total, size):
        chunk = pending[start : start + size]
        names = [_character_name(character) for _, character in chunk]
        try:
            payload = client.chat_json(
                _system_prompt(language),
                _user_prompt(chunk, evidence, language),
                temperature=0.1,
                max_tokens=min(1536, 160 * len(chunk)),
            )
            descriptions = _parse_batch(payload, len(chunk), names)
        except LlmError as exc:
            print(f"[character_profile] LLM descriptions skipped: {exc}")
            descriptions = []
        except Exception as exc:  # a broken client must not break the book
            print(f"[character_profile] LLM descriptions failed: {exc}")
            descriptions = []

        for (_, character), description in zip(chunk, descriptions or []):
            text = _clean_text(description)
            if text:
                character["description"] = text
                character["description_source"] = _SOURCE_LLM

        done += len(chunk)
        if progress is not None:
            progress(done / total, desc=f"Characters {done}/{total}")

    return characters
