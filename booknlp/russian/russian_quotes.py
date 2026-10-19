"""Dialogue (quote) detection and speaker attribution for Russian text.

Russian fiction marks speech in two ways:

* quotation marks: «…», „…“, “…” and plain ASCII quotes;
* an em dash: “— Пойдём, — сказал Иван.”

This module finds both kinds of dialogue spans as plain character offsets and
tries to attribute each span to a speaker by looking at the surrounding text
of the same paragraph (“сказал Иван”, “Иван произнёс:”, “сказал он”, …).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .russian_nlp_utils import is_attribution_verb, is_speech_verb

# opener -> allowed closers (checked in order of preference)
OPENING_QUOTES: dict[str, tuple[str, ...]] = {
    "«": ("»",),
    "„": ("“", "»"),
    "“": ("”",),
    '"': ('"',),
}

DASHES = "—–-"

NAME_RE = re.compile(r"[А-ЯЁЁ][а-яёё]+(?:\s+[А-ЯЁЁ][а-яёё]+){0,2}")

# Function words, pronouns, particles, adverbs and numerals that are commonly
# capitalized only because they start a sentence (or because the small POS/NER
# model mislabels them). A capitalized surface whose tokens all appear here is
# never treated as a character name.
COMMON_CAPITALIZED = {
    # conjunctions and particles
    "и", "а", "но", "да", "ни", "не", "нет", "же", "ли", "бы", "ведь",
    "вот", "вон", "даже", "только", "лишь", "именно", "разве", "неужели",
    "пусть", "пускай", "уж", "либо", "или", "то", "что", "чтобы", "если",
    "хотя", "зато", "притом", "причём", "причем", "однако", "итак",
    "пока", "сейчас", "позже", "рано", "может", "во-первых", "во-вторых",
    # pronouns and pronominal forms
    "я", "мы", "вы", "ты", "он", "она", "оно", "они", "те", "эти", "это",
    "тот", "та", "того", "тому", "той", "тех", "этот", "эта", "этих",
    "этим", "этими", "себя", "себе", "собой", "свой", "своя", "своё",
    "свое", "свои", "мой", "моя", "моё", "мое", "мои", "твой", "твоя",
    "твоё", "твое", "твои", "наш", "наша", "наше", "наши", "ваш", "ваша",
    "ваше", "ваши", "его", "её", "ее", "ему", "ей", "ею", "их", "им",
    "ими", "него", "неё", "нее", "ней", "нему", "нею", "них", "ним",
    "ними", "меня", "тебя", "мне", "тебе", "нам", "вам", "вас", "нас",
    "кто", "кого", "кому", "кем", "ком", "чего", "чему", "чем",
    "какой", "какая", "какое", "какие", "какого", "каких", "каков",
    "какова", "каково", "каковы", "таков", "такова", "таково", "таковы",
    "который", "которая", "которое",
    "которые", "чей", "чья", "чьё", "чье", "чьи", "каждый", "каждая",
    "каждое", "каждые", "любой", "любая", "любое", "любые", "всякий",
    "всякая", "всякое", "всякие", "сам", "сама", "само", "сами", "самый",
    "самая", "самое", "самые", "весь", "вся", "всё", "все", "всех",
    "всем", "всеми", "иной", "иная", "иное", "иные", "другой", "другая",
    "другое", "другие", "никто", "никого", "никому", "никем", "ничто",
    "ничего", "ничему", "ничем", "некто", "некого", "некому", "некем",
    "нечто", "нечего", "нечему", "нечем",
    # adverbs, parentheticals and other sentence openers
    "тут", "там", "здесь", "тогда", "теперь", "всегда", "никогда",
    "иногда", "часто", "редко", "снова", "опять", "вдруг", "сразу",
    "потом", "затем", "сначала", "сперва", "наконец", "уже", "ещё", "еще",
    "весьма", "очень", "слишком", "совсем", "почти", "едва", "чуть",
    "просто", "прямо", "возможно", "вероятно", "наверное", "наверно",
    "конечно", "разумеется", "действительно", "кажется", "пожалуй",
    "видимо", "похоже", "значит", "например", "наоборот", "напротив",
    "кстати", "вообще", "обычно", "короче", "словом", "следовательно",
    "поэтому", "потому", "оттого", "посему", "зачем", "почему", "отчего",
    "где", "куда", "откуда", "когда", "сколько", "насколько", "как",
    "так", "также", "тоже", "почти", "верно", "правда", "довольно",
    "хватит", "пора", "жаль", "стыдно", "надо", "нужно", "можно",
    "нельзя", "лучше", "наверх", "назад", "вперёд", "вперед", "слева",
    "справа", "вдали", "везде", "нигде", "несколько", "внизу", "вверху",
    "впереди", "позади", "клянусь",
    # prepositions and preposition-like
    "без", "в", "во", "для", "до", "за", "из", "к", "ко", "на", "над",
    "о", "об", "обо", "от", "по", "под", "при", "про", "с", "со", "у",
    "через", "между", "вокруг", "перед", "около", "вместо", "кроме",
    "среди", "сквозь", "вдоль", "после", "прежде", "насчёт", "насчет",
    "вроде", "согласно", "благодаря", "вопреки", "несмотря",
    # numerals and ordinals
    "один", "одна", "одно", "два", "две", "три", "четыре", "пять",
    "шесть", "семь", "восемь", "девять", "десять", "двадцать",
    "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят",
    "восемьдесят", "девяносто", "сто", "тысяча", "первый", "первая",
    "первое", "первые", "первого", "первой", "второй", "вторая",
    "второе", "вторые", "третий", "третья", "третье", "третьи",
    "четвёртый", "четвертый", "четвертая", "четвертой", "пятый",
    "шестой", "седьмой", "восьмой", "девятый", "десятый", "десятая",
    "десятой", "сотый", "тысячный",
    # interjections and common speech forms
    "ну", "эй", "ах", "ох", "ух", "ой", "ай", "ого", "увы", "спасибо",
    "пожалуйста", "прости", "простите", "здравствуй", "здравствуйте",
    "привет", "пока", "боже", "господи", "чёрт", "черт", "ладно",
    "хорошо", "давай", "давайте", "посмотри", "послушай", "подожди",
    "погоди", "стой", "тише", "молчи", "вдруг",
}
COMMON_CAPITALIZED = set(COMMON_CAPITALIZED)


# Gendered pronouns and plural pronouns (plural forms do not constrain gender)
PRONOUN_GENDER = {
    "он": "male",
    "его": "male",
    "ему": "male",
    "него": "male",
    "она": "female",
    "её": "female",
    "ее": "female",
    "ей": "female",
    "ею": "female",
    "ней": "female",
    "нее": "female",
    "неё": "female",
}
PLURAL_PRONOUNS = {"они", "их", "них", "ним", "им"}
PRONOUNS = set(PRONOUN_GENDER) | PLURAL_PRONOUNS

MAX_QUOTE_SPAN = 4000  # characters; longer spans are almost certainly broken


@dataclass(frozen=True)
class DialogueSpan:
    """A span of spoken text as character offsets into the book text."""

    start: int
    end: int
    kind: str  # 'quote' («…» style) or 'dash' (—… style)


@dataclass(frozen=True)
class SpeakerHint:
    """Who most likely produced a :class:`DialogueSpan`."""

    kind: str  # 'name' or 'pronoun'
    value: str  # surface name ("Иван Петров") or lowercase pronoun ("он")


def iter_paragraphs(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` offsets of non-empty paragraphs."""
    paragraphs: list[tuple[int, int]] = []
    pos = 0
    for match in re.finditer(r"\n+", text):
        segment = text[pos:match.start()]
        if segment.strip():
            paragraphs.append((pos, match.start()))
        pos = match.end()
    segment = text[pos:]
    if segment.strip():
        paragraphs.append((pos, len(text)))
    return paragraphs


def _quote_spans(text: str) -> list[tuple[int, int, int]]:
    """Find quoted spans. Returns ``(content_start, content_end, kind)``."""
    stack: list[tuple[int, tuple[str, ...]]] = []
    spans: list[tuple[int, int, int]] = []

    for index, char in enumerate(text):
        if char in OPENING_QUOTES:
            stack.append((index, OPENING_QUOTES[char]))
            continue

        if not stack:
            continue

        # A closer matches the innermost open quote, but only if it is one of
        # that opener's allowed closers. ASCII quotes close when they opened.
        top_pos, closers = stack[-1]
        if char in closers:
            stack.pop()
            content_start = top_pos + 1
            content_end = index
            if content_end > content_start and (content_end - content_start) <= MAX_QUOTE_SPAN:
                spans.append((content_start, content_end, "quote"))

    # Keep outermost, non-overlapping spans in reading order.
    spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    kept: list[tuple[int, int, int]] = []
    last_end = -1
    for start, end, kind in spans:
        if start >= last_end:
            kept.append((start, end, kind))
            last_end = end
    return kept


def _is_standalone_dash(text: str, index: int, para_start: int, para_end: int) -> bool:
    """True for dialogue dashes, False for hyphens inside words."""
    if text[index] not in DASHES:
        return False
    before = text[index - 1] if index - 1 >= 0 else ""
    after = text[index + 1] if index + 1 < len(text) else ""
    starts_ok = index - 1 < para_start or before.isspace() or before in "«»\"'"
    ends_ok = index + 1 >= para_end or after.isspace() or after in "«»\"'"
    return starts_ok and ends_ok


def _looks_like_sentence_boundary_before(text: str, index: int, para_start: int) -> bool:
    """True when only sentence-final punctuation precedes the dash."""
    cursor = index - 1
    while cursor >= para_start and text[cursor].isspace():
        cursor -= 1
    if cursor < para_start:
        return True  # dash opens the paragraph
    return text[cursor] in ".!?…»\"“”"


def _segment_is_attribution(segment: str) -> bool:
    """True when a dash segment is stage direction, not speech: “— сказал Иван.”"""
    words = re.findall(r"[А-ЯЁа-яёЁё']+", segment)[:8]
    return any(is_attribution_verb(word) for word in words)


_WORD_IN_SEGMENT_RE = re.compile(r"[А-ЯЁа-яёЁё']+")


def _first_speech_verb_match(segment: str):
    for match in _WORD_IN_SEGMENT_RE.finditer(segment):
        if is_attribution_verb(match.group(0)):
            return match
    return None


def _dash_spans(text: str) -> list[tuple[int, int, int]]:
    """Find em-dash dialogue utterances."""
    spans: list[tuple[int, int, int]] = []
    for para_start, para_end in iter_paragraphs(text):
        dashes = [
            index
            for index in range(para_start, para_end)
            if _is_standalone_dash(text, index, para_start, para_end)
        ]
        if not dashes:
            continue

        for position, dash_index in enumerate(dashes):
            segment_end = dashes[position + 1] if position + 1 < len(dashes) else para_end
            segment = text[dash_index + 1:segment_end]
            if _segment_is_attribution(segment):
                # Inline attribution: “— Пойдём, сказал Иван.” The utterance
                # is what precedes the verb; the verb phrase that follows is
                # picked up as the speaker hint for that utterance.
                verb = _first_speech_verb_match(segment)
                if verb is None or verb.start() < 3:
                    continue
                start = dash_index + 1
                end = dash_index + 1 + verb.start()
                while end > start and (text[end - 1].isspace() or text[end - 1] in ",;:"):
                    end -= 1
                if end - start >= 3:
                    spans.append((start, end, "dash"))
                continue
            # A dash only starts speech after a sentence boundary or at the
            # beginning of a paragraph; "ушёл — домой" is narration.
            if not _looks_like_sentence_boundary_before(text, dash_index, para_start):
                continue
            start = dash_index + 1
            end = segment_end
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if end > start:
                spans.append((start, end, "dash"))
    return spans


def find_dialogue_spans(text: str) -> list[DialogueSpan]:
    """Find all spoken spans (quotes and dash dialogue) in *text*."""
    quotes = _quote_spans(text)
    dash = _dash_spans(text)

    spans: list[DialogueSpan] = [
        DialogueSpan(start=start, end=end, kind=kind) for start, end, kind in quotes
    ]
    quote_ranges = [(start, end) for start, end, kind in quotes]

    for start, end, kind in dash:
        overlaps_quote = any(start < q_end and q_start < end for q_start, q_end in quote_ranges)
        if not overlaps_quote:
            spans.append(DialogueSpan(start=start, end=end, kind=kind))

    spans.sort(key=lambda span: span.start)
    return spans


def _clean_name(candidate: str) -> str | None:
    """Validate a capitalized candidate and return it, or None."""
    words = candidate.split()
    if not words:
        return None
    if any(is_speech_verb(word) for word in words):
        return None
    if all(word.lower() in COMMON_CAPITALIZED for word in words):
        return None
    return candidate


_SENT_BOUNDARY_RE = re.compile(r"[.!?…]+[»\"“”']?\s+(?=[А-ЯЁ«„“—–])")
_WORD_RE = re.compile(r"[А-ЯЁа-яёЁё']+")


def _first_sentence(fragment: str) -> str:
    """Cut *fragment* right after its first sentence end.

    Attribution lives in the same sentence as the utterance («…», — сказал
    Иван.), so anything beyond the first boundary belongs to a new sentence
    and must not leak into the pattern search.
    """
    match = _SENT_BOUNDARY_RE.search(fragment)
    return fragment[: match.end()] if match else fragment


def _last_sentence(fragment: str) -> str:
    matches = list(_SENT_BOUNDARY_RE.finditer(fragment))
    return fragment[matches[-1].end():] if matches else fragment


def _capitalized_name_from(words: list, index: int, direction: int = 1) -> str | None:
    """Collect a run of capitalized words starting at *index* (max 3)."""
    collected = []
    cursor = index
    while 0 <= cursor < len(words) and len(collected) < 3:
        word = words[cursor].group(0)
        if not word or not word[0].isupper():
            break
        collected.append(word)
        cursor += direction
    if not collected:
        return None
    return _clean_name(" ".join(collected))


def _after_patterns(after: str) -> SpeakerHint | None:
    """Speaker written after the utterance: “— сказал Иван.” / “— спросил Иван.”"""
    window = _first_sentence(after)
    words = list(_WORD_RE.finditer(window))
    for index, match in enumerate(words):
        if not is_speech_verb(match.group(0)):
            continue
        # Look ahead: adverb ("тихо") before the name/pronoun.
        for lookahead in range(index + 1, min(index + 4, len(words))):
            word = words[lookahead].group(0)
            if word[0].isupper():
                name = _capitalized_name_from(words, lookahead)
                if name:
                    return SpeakerHint("name", name)
            elif word.lower() in PRONOUNS:
                return SpeakerHint("pronoun", word.lower())
        # Look behind: “Иван сказал…” / “Иван Петров произнёс…”.
        if index > 0:
            word = words[index - 1].group(0)
            if word and word[0].isupper():
                name = _capitalized_name_from(words, index - 1, direction=-1)
                if name:
                    return SpeakerHint("name", name)
        break  # only the first speech verb of the sentence attributes
    return None


def _before_patterns(before: str) -> SpeakerHint | None:
    """Speaker written before the utterance: “Иван произнёс: «…»”."""
    window = _last_sentence(before.rstrip())
    words = list(_WORD_RE.finditer(window))
    if not words:
        return None
    tail = window[words[-1].end():].strip()

    last_word = words[-1].group(0)
    if is_speech_verb(last_word) and tail in {"", ":", "-", "—", "–"}:
        # verb right before the quote: “Иван произнёс:” / “он сказал —”
        if len(words) >= 2:
            previous = words[-2].group(0)
            if previous[0].isupper():
                name = _capitalized_name_from(words, len(words) - 2, direction=-1)
                if name:
                    return SpeakerHint("name", name)
            if previous.lower() in PRONOUNS:
                return SpeakerHint("pronoun", previous.lower())
    else:
        # “Иван сказал задумчиво: «…»” — one adverb between verb and quote.
        for offset in range(1, min(3, len(words))):
            verb_index = len(words) - 1 - offset
            if not is_speech_verb(words[verb_index].group(0)):
                continue
            if not all(
                words[t].group(0)[0].islower()
                for t in range(verb_index + 1, len(words))
            ):
                continue
            if verb_index >= 1:
                previous = words[verb_index - 1].group(0)
                if previous[0].isupper():
                    name = _capitalized_name_from(words, verb_index - 1, direction=-1)
                    if name:
                        return SpeakerHint("name", name)
                if previous.lower() in PRONOUNS:
                    return SpeakerHint("pronoun", previous.lower())
            break

    if tail == ":" and last_word[0].isupper():
        # “Иван: «…»”
        name = _capitalized_name_from(words, len(words) - 1, direction=-1)
        if name:
            return SpeakerHint("name", name)

    return None


def find_speaker_hint(text: str, span: DialogueSpan, para_start: int, para_end: int) -> SpeakerHint | None:
    """Guess the speaker of *span* from the same paragraph's text."""
    after = text[span.end:para_end][:140]
    before = text[para_start:span.start][-140:]

    hint = _after_patterns(after)
    if hint:
        return hint
    return _before_patterns(before)
