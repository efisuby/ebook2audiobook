"""Russian book analysis producing BookNLP-compatible output files.

The English pipeline (``booknlp/english``) relies on models trained for
English, so Russian text is processed here instead. The output files use the
same names and column/JSON layout as the English pipeline, which lets
``sml_extractor`` consume either language without special cases:

* ``<id>.tokens``   -- token table (``token_ID_within_document``, ``paragraph_ID``, ``word``, ...)
* ``<id>.quotes``   -- dialogue spans (``quote_start``, ``quote_end``, ``char_id``)
* ``<id>.entities`` -- person mentions (``COREF start end prop cat text``)
* ``<id>.book``     -- JSON with characters, gender and name mentions
* ``<id>.characters_simple.json`` -- the GUI/CLI character list

Pipeline outline:

1. tokenize with ``ru_core_news_sm`` (regex fallback when spaCy is absent);
2. find dialogue: «...»/„…“ quotes and em-dash dialogue
   (:mod:`booknlp.russian.russian_quotes`);
3. attribute each span to a speaker via "сказал Иван", "Иван произнёс:",
   "сказал он" patterns;
4. cluster person names (NER + recurring capitalized sequences + attribution
   names) sharing a first/last name part;
5. resolve pronouns to clusters with BERT embeddings
   (:mod:`booknlp.russian.russian_coref`), with a lexical fallback;
6. write BookNLP-compatible output files.
"""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .russian_coref import CorefCandidate, RussianCorefResolver
from .russian_nlp_utils import is_speech_verb, load_russian_spacy_model
from .russian_quotes import (
    COMMON_CAPITALIZED,
    NAME_RE,
    PLURAL_PRONOUNS,
    PRONOUN_GENDER,
    DialogueSpan,
    SpeakerHint,
    find_dialogue_spans,
    find_speaker_hint,
    iter_paragraphs,
)

_FALLBACK_TOKEN_RE = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]", re.UNICODE)
_SENTENCE_END_RE = re.compile(r"[.!?…]+[»\"“”']?\s+")
_PLAIN_NAME_RE = re.compile(r"[А-ЯЁа-яёЁёA-Za-z]+")

AGE_TERMS = {
    "child": {
        "ребёнок", "ребенок", "мальчик", "девочка", "дитя", "младенец",
        "малыш", "сынок", "дочь", "кадет",
    },
    "teen": {"подросток", "подростковый", "отрок", "гимназист", "лицеист"},
    "elder": {
        "старик", "старуха", "бабушка", "дедушка", "пожилой", "дряхлый",
        "старый", "старая", "дед", "бабка",
    },
}

# Gender tables kept in sync with sml_extractor/core.py.
_MALE_SUFFIXES = ("ов", "ев", "ёв", "ин", "ын", "ий", "ый", "ой", "ь", "й")
_FEMALE_SUFFIXES = ("а", "я", "ия", "ая", "ья", "на", "ина")
_MALE_NAMES = {
    "иван", "пётр", "петр", "александр", "дмитрий", "сергей", "николай",
    "андрей", "павел", "игорь", "владимир", "алексей", "роман", "максим",
    "артём", "артем", "юрий", "борис", "виктор", "виталий", "георгий",
    "григорий", "денис", "евгений", "захар", "илья", "кирилл", "леонид",
    "михаил", "олег", "станислав", "степан", "тимур", "фёдор", "федор",
    "филипп", "ярослав",
}
_FEMALE_NAMES = {
    "анна", "мария", "елена", "наталья", "ольга", "татьяна", "ксения",
    "дарья", "марина", "светлана", "вера", "екатерина", "анастасия",
    "алина", "полина", "юлия", "ирина", "лариса", "милана", "софия",
    "алиса", "варвара", "валерия", "кристина", "надежда", "людмила",
    "галя", "лидия", "зоя", "тамара", "раиса", "нина", "яна", "рина",
}


@dataclass
class _Token:
    word: str
    start: int
    end: int
    paragraph: int
    sentence: int
    lemma: str = ""
    pos: str = ""
    ner: str = ""


@dataclass
class _NameMention:
    surface: str
    tokens: tuple
    start: int
    end: int
    sentence: int
    source: str  # 'ner', 'attribution' or 'recurring'


@dataclass
class _Cluster:
    key: tuple
    surfaces: Counter = field(default_factory=Counter)
    mentions: list = field(default_factory=list)
    pronouns: Counter = field(default_factory=Counter)
    name_gender: Optional[str] = None
    final_id: Optional[int] = None

    @property
    def canonical(self) -> str:
        if not self.surfaces:
            return " ".join(self.key).title()
        return max(
            self.surfaces.items(), key=lambda item: (item[1], len(item[0]))
        )[0].title()

    @property
    def gender(self) -> Optional[str]:
        if self.name_gender:
            return self.name_gender
        male = self.pronouns.get("он", 0)
        female = self.pronouns.get("она", 0)
        if male > female:
            return "male"
        if female > male:
            return "female"
        return None


class _OffsetIndex:
    """Bisect-based lookup of paragraph/sentence index for a char offset."""

    def __init__(self, spans: list):
        self.spans = spans
        self.starts = [start for start, _ in spans]

    def index_at(self, offset: int) -> int:
        if not self.starts:
            return 0
        return max(0, bisect_right(self.starts, offset) - 1)

    def span_at(self, offset: int) -> tuple:
        if not self.spans:
            return (0, 0)
        return self.spans[self.index_at(offset)]


class RussianBookNLP:
    """Russian counterpart of ``EnglishBookNLP``."""

    def __init__(self, language: str = "rus", model_params: Optional[Dict] = None):
        """Initialize the pipeline.

        Args:
            language: Language code; only ``'rus'`` is supported.
            model_params: BookNLP-style parameters. Recognized keys:
                ``pipeline`` (ignored, API parity), ``model`` (``small``/``big``,
                both map to ``ru_core_news_sm``), ``model_path`` (ignored; the
                spaCy model is loaded from the active environment).
        """
        self.language = language
        self.model_params = dict(model_params or {})
        self.model_params.setdefault("model", "small")
        self.nlp = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def process(
        self,
        input_file: str,
        output_dir: str,
        book_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze *input_file* and write BookNLP-compatible output files."""
        text = Path(input_file).read_text(encoding="utf-8")
        if not book_id:
            book_id = Path(input_file).stem
        out_dir = Path(output_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

        print("--- loading Russian spaCy model ---")
        self.nlp = load_russian_spacy_model()
        if self.nlp is None:
            print(
                "Warning: ru_core_news_sm is not installed; falling back to "
                "regex tokenization and no NER. Install it with: "
                'uv pip install "$(python -m spacy info ru_core_news_sm --url)"'
            )

        doc = self.nlp(text) if self.nlp is not None else None

        para_index = _OffsetIndex(iter_paragraphs(text))
        sent_index = _OffsetIndex(_sentence_spans(text, doc))
        tokens = _build_tokens(text, para_index, sent_index, doc)
        print(f"--- tokenized {len(tokens)} tokens, {len(para_index.spans)} paragraphs ---")

        name_mentions = self._collect_name_mentions(text, sent_index, doc)
        spans = find_dialogue_spans(text)
        hints = [self._hint_for(text, span, para_index) for span in spans]

        # Names discovered in attributions ("сказал Иван") are strong
        # character evidence, so they join the mention pool before clustering.
        for span, hint in zip(spans, hints):
            if hint is not None and hint.kind == "name":
                hint_tokens = _name_tokens(hint.value)
                if not hint_tokens or not self._plausible_name(hint_tokens):
                    continue
                name_mentions.append(
                    _NameMention(
                        surface=hint.value,
                        tokens=hint_tokens,
                        start=span.start,
                        end=span.end,
                        sentence=sent_index.index_at(span.start),
                        source="attribution",
                    )
                )

        clusters = _cluster_names(name_mentions)
        resolver = RussianCorefResolver()
        print(
            f"--- {len(clusters)} name clusters, {len(spans)} dialogue spans, "
            f"coref backend: {resolver.backend} ---"
        )

        span_clusters = self._attribute_spans(
            text, spans, para_index, sent_index, clusters, resolver, hints
        )
        self._resolve_pronouns(clusters, tokens, text, sent_index, resolver)
        characters, span_char_ids = self._build_characters(
            clusters, span_clusters, text, sent_index
        )

        self._write_outputs(
            out_dir, book_id, tokens, spans, span_char_ids, characters, clusters
        )
        print(f"--- wrote outputs for {len(characters)} characters ---")

        return {
            "book_id": book_id,
            "output_dir": str(out_dir),
            "characters": characters,
        }

    # ------------------------------------------------------------------
    # mention discovery
    # ------------------------------------------------------------------
    def _collect_name_mentions(
        self, text: str, sent_index: _OffsetIndex, doc: Any
    ) -> List[_NameMention]:
        mentions: List[_NameMention] = []
        seen: set = set()

        if doc is not None:
            for ent in doc.ents:
                if ent.label_ != "PER" or not ent.text.strip():
                    continue
                surface = ent.text.strip()
                tokens = _name_tokens(surface)
                if not tokens or not self._plausible_name(tokens):
                    continue
                if self._has_verb_pos(doc, ent.start_char, ent.end_char):
                    continue
                mentions.append(
                    _NameMention(
                        surface=surface,
                        tokens=tokens,
                        start=ent.start_char,
                        end=ent.start_char + len(surface),
                        sentence=sent_index.index_at(ent.start_char),
                        source="ner",
                    )
                )
                seen.add((ent.start_char, ent.start_char + len(surface)))

        # Recurring capitalized sequences catch names the small NER misses.
        # ``ru_core_news_sm`` capitalizes every sentence-initial word, so a
        # capitalized token is only trusted as a name when it repeats and at
        # least two occurrences sit mid-sentence (a sentence-initial capital is
        # usually just the first word of a sentence), and at least one of those
        # mid-sentence occurrences is tagged PROPN or falls inside a PER
        # entity. This drops the "кто/то/ну" style noise the POS model lets
        # through (see ``COMMON_CAPITALIZED`` for the word-level guard).
        per_spans: List[tuple] = []
        if doc is not None:
            per_spans = sorted(
                (ent.start_char, ent.start_char + len(ent.text))
                for ent in doc.ents
                if ent.label_ == "PER" and ent.text.strip()
            )
        per_cursor = 0

        candidates = []
        mid_counts: Counter = Counter()
        mid_proper: set = set()
        for match in NAME_RE.finditer(text):
            surface = match.group(0)
            start, end = match.start(), match.end()
            tokens = _name_tokens(surface)
            if not tokens or not self._plausible_name(tokens):
                continue
            if (start, end) in seen:
                continue
            if self._has_verb_pos(doc, start, end):
                continue

            while per_cursor < len(per_spans) and per_spans[per_cursor][1] <= start:
                per_cursor += 1
            in_per = per_cursor < len(per_spans) and per_spans[per_cursor][0] < end

            mid = not self._starts_sentence(text, doc, start)
            candidates.append((surface, tokens, start, end))
            if mid:
                mid_counts[tokens] += 1
                if in_per or self._is_proper_pos(doc, start, end):
                    mid_proper.add(tokens)

        eligible = {
            tokens for tokens, count in mid_counts.items() if count >= 2
        } & mid_proper

        for surface, tokens, start, end in candidates:
            if tokens not in eligible:
                continue
            mentions.append(
                _NameMention(
                    surface=surface,
                    tokens=tokens,
                    start=start,
                    end=end,
                    sentence=sent_index.index_at(start),
                    source="recurring",
                )
            )

        mentions.sort(key=lambda mention: mention.start)
        return mentions

    @staticmethod
    def _has_verb_pos(doc: Any, start: int, end: int) -> bool:
        """True when spaCy tags any token in ``text[start:end]`` as a verb.

        ``ru_core_news_sm`` NER sometimes labels imperatives such as «Пойдём»
        (lemma «пойти», POS VERB) as PER. The POS data is reliable, so it
        vetoes those false character names.
        """
        if doc is None:
            return False
        try:
            span = doc.char_span(start, end, alignment_mode="expand")
        except Exception:
            span = None
        if span is None:
            return False
        return any(token.pos_ in {"VERB", "AUX"} for token in span)

    @staticmethod
    def _starts_sentence(text: str, doc: Any, start: int) -> bool:
        """True when the character at *start* begins a sentence.

        Uses spaCy's ``is_sent_start`` when the model is available and falls
        back to the punctuation preceding *start* otherwise.
        """
        if doc is not None:
            try:
                span = doc.char_span(start, start + 1, alignment_mode="expand")
            except Exception:
                span = None
            if span is not None and len(span):
                return bool(span[0].is_sent_start)
        index = start - 1
        while index >= 0 and text[index] in " \t\r\n":
            index -= 1
        if index < 0:
            return True
        return text[index] in ".!?…"

    @staticmethod
    def _is_proper_pos(doc: Any, start: int, end: int) -> bool:
        """True when spaCy tags every token in ``text[start:end]`` as PROPN."""
        if doc is None:
            return False
        try:
            span = doc.char_span(start, end, alignment_mode="expand")
        except Exception:
            span = None
        if span is None or not len(span):
            return False
        return all(token.pos_ == "PROPN" for token in span)

    @staticmethod
    def _plausible_name(tokens: tuple) -> bool:
        if any(token in COMMON_CAPITALIZED for token in tokens):
            return False
        if any(is_speech_verb(token) for token in tokens):
            return False
        if any(len(token) <= 1 for token in tokens):
            return False
        return True

    # ------------------------------------------------------------------
    # dialogue attribution
    # ------------------------------------------------------------------
    @staticmethod
    def _hint_for(text: str, span: DialogueSpan, para_index: _OffsetIndex) -> Optional[SpeakerHint]:
        para_start, para_end = para_index.span_at(span.start)
        return find_speaker_hint(text, span, para_start, para_end)

    @staticmethod
    def _attribute_spans(
        text: str,
        spans: list,
        para_index: "_OffsetIndex",
        sent_index: _OffsetIndex,
        clusters: list,
        resolver: RussianCorefResolver,
        hints: list,
    ) -> Dict[int, Optional[_Cluster]]:
        """Map dialogue span index -> speaking cluster (or None = narrator).

        A span without its own attribution inherits the speaker of the
        preceding attributed span in the same paragraph, which matches how
        multi-turn dialogue is laid out in Russian fiction.
        """
        surface_index: Dict[tuple, _Cluster] = {}
        for cluster in clusters:
            surface_index[cluster.key] = cluster
            for surface in cluster.surfaces:
                surface_index[_name_tokens(surface.title())] = cluster

        span_clusters: Dict[int, Optional[_Cluster]] = {}
        paragraph_speakers: Dict[int, _Cluster] = {}

        for index, span in enumerate(spans):
            para_id = para_index.index_at(span.start)
            sent_idx = sent_index.index_at(span.start)
            cluster: Optional[_Cluster] = None
            hint = hints[index]

            if hint is not None:
                if hint.kind == "name":
                    cluster = _find_cluster(surface_index, clusters, hint.value)
                else:
                    candidates = _coref_candidates(
                        clusters, sent_idx, text, sent_index, before_offset=span.start
                    )
                    resolved = resolver.resolve(
                        hint.value, _sentence_text(text, sent_index, sent_idx), candidates
                    )
                    cluster = _cluster_by_ref(clusters, resolved)

            if cluster is None:
                cluster = paragraph_speakers.get(para_id)
            else:
                paragraph_speakers[para_id] = cluster

            span_clusters[index] = cluster

        return span_clusters

    # ------------------------------------------------------------------
    # pronouns (BERT coreference)
    # ------------------------------------------------------------------
    def _resolve_pronouns(
        self,
        clusters: list,
        tokens: list,
        text: str,
        sent_index: _OffsetIndex,
        resolver: RussianCorefResolver,
    ) -> None:
        for token in tokens:
            pronoun = token.word.lower()
            if pronoun not in PRONOUN_GENDER and pronoun not in PLURAL_PRONOUNS:
                continue
            candidates = _coref_candidates(
                clusters, token.sentence, text, sent_index, before_offset=token.start
            )
            if not candidates:
                continue
            resolved = resolver.resolve(
                pronoun, _sentence_text(text, sent_index, token.sentence), candidates
            )
            cluster = _cluster_by_ref(clusters, resolved)
            if cluster is not None:
                cluster.pronouns[pronoun] += 1

    # ------------------------------------------------------------------
    # character assembly
    # ------------------------------------------------------------------
    def _build_characters(
        self,
        clusters: list,
        span_clusters: Dict[int, Optional[_Cluster]],
        text: str,
        sent_index: _OffsetIndex,
    ) -> tuple:
        """Build the .book character list and span index -> final char id map."""
        speaking = {id(cluster) for cluster in span_clusters.values() if cluster}
        kept = [
            cluster
            for cluster in clusters
            if len(cluster.mentions) >= 2 or id(cluster) in speaking
        ]
        kept.sort(
            key=lambda cluster: min((m.start for m in cluster.mentions), default=0)
        )
        for number, cluster in enumerate(kept, start=1):
            cluster.final_id = number

        characters = []
        for cluster in kept:
            gender = cluster.gender or _gender_from_name(cluster.canonical) or "unknown"
            characters.append(
                {
                    "id": cluster.final_id,
                    "count": sum(cluster.surfaces.values()),
                    "g": None if gender == "unknown" else gender,
                    "age": _age_for_cluster(cluster, text, sent_index),
                    "mentions": {
                        "proper": [
                            {"c": count, "n": surface.title()}
                            for surface, count in cluster.surfaces.most_common()
                        ],
                        "common": [],
                        "pronoun": [
                            {"c": count, "n": pronoun}
                            for pronoun, count in cluster.pronouns.most_common()
                        ],
                    },
                    "modifiers": [],
                }
            )

        span_char_ids: Dict[int, Optional[int]] = {}
        for index, cluster in span_clusters.items():
            span_char_ids[index] = cluster.final_id if cluster is not None else None
        return characters, span_char_ids

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def _write_outputs(
        self,
        out_dir: Path,
        book_id: str,
        tokens: list,
        spans: list,
        span_char_ids: Dict[int, Optional[int]],
        characters: list,
        clusters: list,
    ) -> None:
        # .tokens
        rows = []
        for index, token in enumerate(tokens):
            values = (
                index,
                token.sentence,
                token.paragraph,
                token.word,
                token.lemma,
                token.pos,
                token.ner,
                token.start,
                token.end,
            )
            rows.append(
                "\t".join(
                    str(value).replace("\t", " ").replace("\n", " ")
                    for value in values
                )
            )
        (out_dir / f"{book_id}.tokens").write_text(
            "token_ID_within_document\tsentence_ID\tparagraph_ID\tword\tlemma\tPOS"
            "\tNER\tchar_start\tchar_end\n" + "\n".join(rows) + "\n",
            encoding="utf-8",
        )

        # .quotes -- token-indexed dialogue spans
        token_starts = [token.start for token in tokens]
        token_ends = [token.end for token in tokens]
        quote_lines = []
        for index, span in enumerate(spans):
            # First token *overlapping* the span: spans that start just after
            # a dash (between tokens) must not pull the dash itself in.
            first = max(0, bisect_right(token_ends, span.start))
            last = max(first, bisect_right(token_starts, span.end - 1) - 1)
            if last < first or first >= len(tokens):
                continue
            char_id = span_char_ids.get(index)
            quote_lines.append(f"{first}\t{last}\t{'None' if char_id is None else char_id}")
        (out_dir / f"{book_id}.quotes").write_text(
            "quote_start\tquote_end\tchar_id\n"
            + "\n".join(quote_lines)
            + ("\n" if quote_lines else ""),
            encoding="utf-8",
        )

        # .entities
        entity_lines = ["COREF\tstart_token\tend_token\tprop\tcat\ttext"]
        for cluster in clusters:
            if cluster.final_id is None:
                continue
            for mention in cluster.mentions:
                first = max(0, bisect_right(token_ends, mention.start))
                last = max(first, bisect_right(token_starts, mention.end - 1) - 1)
                if last < first or first >= len(tokens):
                    continue
                value = mention.surface.replace("\t", " ").replace("\n", " ")
                entity_lines.append(f"{cluster.final_id}\t{first}\t{last}\tPER\tPER\t{value}")
        (out_dir / f"{book_id}.entities").write_text(
            "\n".join(entity_lines) + "\n", encoding="utf-8"
        )

        # .book
        book_characters = [
            {key: value for key, value in character.items() if key != "age"}
            for character in characters
        ]
        (out_dir / f"{book_id}.book").write_text(
            json.dumps({"characters": book_characters}, ensure_ascii=False),
            encoding="utf-8",
        )

        # .characters_simple.json
        simple_characters = [
            {
                "normalized_name": "Narrator",
                "inferred_gender": "unknown",
                "inferred_age_category": "unknown",
                "tts_engine": "XTTSv2-Russian",
                "language": "rus",
                "voice": None,
            }
        ]
        for character in characters:
            proper = character["mentions"]["proper"]
            canonical = proper[0]["n"] if proper else f"Character{character['id']}"
            simple_characters.append(
                {
                    "normalized_name": _camel(canonical),
                    "inferred_gender": character["g"] or "unknown",
                    "inferred_age_category": character.get("age", "adult"),
                    "tts_engine": "XTTSv2-Russian",
                    "language": "rus",
                    "voice": None,
                }
            )
        (out_dir / f"{book_id}.characters_simple.json").write_text(
            json.dumps({"characters": simple_characters}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _camel(name: str) -> str:
    """CamelCase a name exactly like ``sml_extractor.sml_generator`` does."""
    cleaned = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE).replace("_", "")
    parts = cleaned.strip().split()
    return "".join(word.capitalize() for word in parts) if parts else "Unknown"


def _gender_from_name(name: str) -> Optional[str]:
    lowered = name.lower().strip()
    if not lowered:
        return None
    if lowered in _FEMALE_NAMES:
        return "female"
    if lowered in _MALE_NAMES:
        return "male"
    for suffix in _FEMALE_SUFFIXES:
        if lowered.endswith(suffix):
            return "female"
    for suffix in _MALE_SUFFIXES:
        if lowered.endswith(suffix):
            return "male"
    return None


def _name_tokens(surface: str) -> tuple:
    """Split a name into lowercase word tokens: "Иван Петров" -> ("иван","петров")."""
    return tuple(token.lower() for token in _PLAIN_NAME_RE.findall(surface))


def _sentence_spans(text: str, doc: Any) -> list:
    if doc is not None:
        return [(sent.start_char, sent.end_char) for sent in doc.sents]
    spans = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(text):
        if match.end() > start:
            spans.append((start, match.end()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans or [(0, len(text))]


def _sentence_text(text: str, sent_index: _OffsetIndex, index: int) -> str:
    if index >= len(sent_index.spans):
        return ""
    start, end = sent_index.spans[index]
    return text[start:end]


def _build_tokens(
    text: str, para_index: _OffsetIndex, sent_index: _OffsetIndex, doc: Any
) -> List[_Token]:
    tokens: List[_Token] = []

    if doc is not None:
        for token in doc:
            if token.is_space or not token.text.strip():
                continue
            offset = token.idx
            tokens.append(
                _Token(
                    word=token.text,
                    start=offset,
                    end=offset + len(token.text),
                    paragraph=para_index.index_at(offset),
                    sentence=sent_index.index_at(offset),
                    lemma=token.lemma_,
                    pos=token.pos_,
                    ner=token.ent_type_ or "",
                )
            )
        return tokens

    for match in _FALLBACK_TOKEN_RE.finditer(text):
        offset = match.start()
        tokens.append(
            _Token(
                word=match.group(0),
                start=offset,
                end=match.end(),
                paragraph=para_index.index_at(offset),
                sentence=sent_index.index_at(offset),
            )
        )
    return tokens


def _cluster_names(mentions: List[_NameMention]) -> List[_Cluster]:
    """Group name surfaces sharing a first/last name part (union-find)."""
    by_surface: Dict[tuple, _Cluster] = {}
    token_index: Dict[str, set] = defaultdict(set)

    for mention in mentions:
        if not mention.tokens:
            continue
        cluster = by_surface.get(mention.tokens)
        if cluster is None:
            cluster = _Cluster(key=mention.tokens)
            by_surface[mention.tokens] = cluster
        cluster.surfaces[mention.surface] += 1
        cluster.mentions.append(mention)
        if cluster.name_gender is None:
            cluster.name_gender = _gender_from_name(mention.surface)
        for token in mention.tokens:
            token_index[token].add(mention.tokens)

    # A token only links surfaces when it also occurs inside a multi-word
    # name: "Иван" links "Иван" <-> "Иван Петров", but a lone frequent word
    # never merges unrelated single-word names.
    linking = {
        token
        for token, surfaces in token_index.items()
        if any(len(surface) > 1 for surface in surfaces)
    }

    parent = {surface: surface for surface in by_surface}

    def find(item: tuple) -> tuple:
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != item:
            parent[item], item = root, parent[item]
        return root

    def union(left: tuple, right: tuple) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for token in linking:
        surfaces = sorted(token_index[token])
        for other in surfaces[1:]:
            union(surfaces[0], other)

    grouped: Dict[tuple, list] = defaultdict(list)
    for surface in by_surface:
        grouped[find(surface)].append(surface)

    clusters: List[_Cluster] = []
    for members in grouped.values():
        if len(members) == 1:
            clusters.append(by_surface[members[0]])
            continue
        merged = _Cluster(key=members[0])
        for member in members:
            source = by_surface[member]
            merged.surfaces.update(source.surfaces)
            merged.mentions.extend(source.mentions)
            merged.pronouns.update(source.pronouns)
            if merged.name_gender is None:
                merged.name_gender = source.name_gender
        clusters.append(merged)

    clusters.sort(
        key=lambda cluster: min((m.start for m in cluster.mentions), default=0)
    )
    return clusters


def _find_cluster(
    surface_index: Dict[tuple, _Cluster], clusters: list, name: str
) -> Optional[_Cluster]:
    """Look up a cluster by a (possibly partial) speaker name."""
    tokens = _name_tokens(name)
    if not tokens:
        return None
    cluster = surface_index.get(tokens)
    if cluster is not None:
        return cluster
    token_set = set(tokens)
    # "Иван" -> "Иван Петров" (hint is a part of the full name)
    for candidate in clusters:
        if token_set < set(candidate.key):
            return candidate
    # "Петров" -> "Иван Петров" (hint is one word of a longer surface)
    for candidate in clusters:
        if any(token_set <= set(surface) and token_set != set(surface)
               for surface in candidate.surfaces):
            return candidate
    return None


def _cluster_by_ref(clusters: list, ref) -> Optional[_Cluster]:
    if ref is None:
        return None
    for cluster in clusters:
        if id(cluster) == ref:
            return cluster
    return None


def _coref_candidates(
    clusters: list,
    sentence_index: int,
    text: str,
    sent_index: _OffsetIndex,
    window: int = 5,
    before_offset: Optional[int] = None,
) -> list:
    """Character clusters mentioned in the current or preceding sentences.

    ``before_offset`` enforces the coreference constraint that an antecedent
    must appear before the pronoun: mentions starting at/after that character
    offset are not candidates (prevents "— Пойдём... — сказал он" from
    resolving "он" to the word inside the quoted speech itself).
    """
    candidates = []
    for cluster in clusters:
        recent = [
            mention.sentence
            for mention in cluster.mentions
            if 0 <= sentence_index - mention.sentence <= window
            and (before_offset is None or mention.start < before_offset)
        ]
        if not recent:
            continue
        distance = sentence_index - min(recent)
        contexts = []
        seen_sentences = set()
        for sentence in sorted(recent, reverse=True):
            if sentence in seen_sentences:
                continue
            seen_sentences.add(sentence)
            contexts.append(_sentence_text(text, sent_index, sentence))
            if len(contexts) >= 3:
                break
        candidates.append(
            CorefCandidate(
                cluster_id=id(cluster),
                sentences=contexts,
                distance=max(distance, 0),
                gender=cluster.gender,
                mentions=len(cluster.mentions),
                pronouns=dict(cluster.pronouns),
            )
        )
    return candidates


def _age_for_cluster(
    cluster: _Cluster, text: str, sent_index: _OffsetIndex
) -> str:
    scores: Counter = Counter()
    for mention in cluster.mentions[:20]:
        start, end = sent_index.span_at(mention.start)
        window = text[start:end].lower()
        for category, terms in AGE_TERMS.items():
            if any(term in window for term in terms):
                scores[category] += 1
    if scores:
        return scores.most_common(1)[0][0]
    return "adult"
