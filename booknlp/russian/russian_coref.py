"""BERT-based coreference helpers for Russian text.

No supervised Russian coreference model is published for transformers, so the
resolver combines a lexical baseline with contextual BERT embeddings
(``sentence-transformers``, a BERT-family encoder):

* every character cluster gets representative sentences;
* each pronoun occurrence (он/она/они/его/её/…) is matched against the
  clusters mentioned in the current and preceding sentences;
* gender agreement filters the candidates, and cosine similarity between the
  pronoun's sentence and the candidates' sentences picks the winner;
* without the model (offline, disabled, download failure) the resolver falls
  back to the nearest preceding mention of matching gender.

Configuration:
    E2A_SML_COREF_MODEL   sentence-transformers model name
                          (default: paraphrase-multilingual-MiniLM-L12-v2)
                          set to "none"/"off" to disable the BERT stage
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_COREF_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# A fresh resolver is built for every ``process`` call, but the encoder itself
# is expensive to load (several seconds) and stateless. One instance per model
# name is shared for the whole process; the per-book embedding cache lives on
# the resolver, so books never share vectors.
_MODEL_CACHE: dict = {}


def _load_shared_model(model_name: str):
    """Return a process-wide ``SentenceTransformer`` for *model_name*.

    ``None`` is cached too, so an unavailable model (offline, missing package,
    bad model id) is only attempted once instead of on every book.  When the
    GPU is full — a local LLM holding most of its memory is the common case —
    the model is retried on the CPU, which is far better than dropping to the
    lexical fallback for a whole book.
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    try:
        from sentence_transformers import SentenceTransformer

        try:
            model: Any = SentenceTransformer(model_name)
        except Exception as gpu_exc:
            print(
                f"[rus-coref] BERT model '{model_name}' could not start on the "
                f"GPU ({gpu_exc}); retrying on the CPU."
            )
            model = SentenceTransformer(model_name, device="cpu")
    except Exception as exc:  # offline, missing package, bad model id, ...
        print(
            f"[rus-coref] BERT model '{model_name}' unavailable "
            f"({exc}); using lexical coreference fallback."
        )
        model = None

    _MODEL_CACHE[model_name] = model
    return model

# keep in sync with russian_quotes.PRONOUN_GENDER / PLURAL_PRONOUNS
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

# Below this cosine similarity the embedding evidence is considered absent.
SIMILARITY_FLOOR = 0.45
# Gender-filtered candidates are required; when a cluster's gender is unknown
# it stays eligible for any pronoun.


def configured_coref_model() -> str | None:
    """Return the BERT model to use, or None when the stage is disabled."""
    raw = os.environ.get("E2A_SML_COREF_MODEL", DEFAULT_COREF_MODEL).strip()
    if raw.lower() in {"", "none", "off", "0", "false"}:
        return None
    return raw


@dataclass
class CorefCandidate:
    """A character cluster that a pronoun could refer to."""

    cluster_id: int
    sentences: list[str]  # representative contexts with this cluster mentioned
    distance: int  # sentences between the pronoun and the nearest mention
    gender: str | None = None
    mentions: int = 1
    pronouns: dict = field(default_factory=dict)


class RussianCorefResolver:
    """Resolves Russian pronouns to character clusters (BERT + lexical)."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name if model_name is not None else configured_coref_model()
        self._model = None
        self._loaded = False
        self._failed = False
        self._cache: dict[str, object] = {}

    @property
    def backend(self) -> str:
        if not self.model_name:
            return "lexical (disabled)"
        if not self._loaded and not self._failed:
            self._ensure_model()
        return f"bert ({self.model_name})" if self._model else "lexical"

    def _ensure_model(self) -> None:
        if self._loaded or self._failed or not self.model_name:
            return
        self._loaded = True
        self._model = _load_shared_model(self.model_name)
        if self._model is None:
            self._failed = True

    def embed(self, texts: list[str]):
        """Return normalized embedding rows, or None when unavailable."""
        if not self.model_name or self._failed:
            return None
        self._ensure_model()
        if self._model is None:
            return None

        missing = [text for text in texts if text not in self._cache]
        if missing:
            try:
                vectors = self._model.encode(
                    list(dict.fromkeys(missing)),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                for text, vector in zip(dict.fromkeys(missing), vectors):
                    self._cache[text] = vector
            except Exception as exc:
                self._failed = True
                print(f"[rus-coref] embedding failed ({exc}); using lexical fallback.")
                return None

        import numpy as np

        return np.stack([self._cache[text] for text in texts])

    def resolve(
        self,
        pronoun: str,
        sentence_text: str,
        candidates: list[CorefCandidate],
    ) -> int | None:
        """Pick the cluster *pronoun* most likely refers to."""
        if not candidates:
            return None

        gender = PRONOUN_GENDER.get(pronoun.lower())
        if gender:
            gendered = [c for c in candidates if c.gender in (None, gender)]
            # Only enforce the filter when it leaves somebody standing.
            if gendered:
                candidates = gendered

        nearest = min(candidates, key=lambda c: c.distance)

        usable = [(c, c.sentences[0]) for c in candidates if c.sentences]
        if not usable:
            return nearest.cluster_id

        vectors = self.embed([sentence_text] + [rep for _, rep in usable])
        if vectors is None:
            return nearest.cluster_id

        import numpy as np

        query = vectors[0]
        best_id = nearest.cluster_id
        best_score = -np.inf
        for index, (candidate, _rep) in enumerate(usable, start=1):
            row = vectors[index]
            similarity = float(np.dot(query, row))
            recency = 0.03 / (1.0 + candidate.distance)
            score = similarity + recency
            if score > best_score:
                best_score = score
                best_id = candidate.cluster_id

        if best_score < SIMILARITY_FLOOR:
            return nearest.cluster_id
        return best_id
