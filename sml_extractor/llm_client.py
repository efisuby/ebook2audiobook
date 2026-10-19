"""In-process LLM backend (transformers) for the optional AI features.

The local model does three jobs: parsing the pasted key-character list, casting
voices from short character descriptions, and labelling the emotion of every
spoken line.  It is loaded on first use with ``transformers`` and kept in
memory for the process, so the GUI needs no external server.

Everything here is optional.  When ``torch``/``transformers`` are missing, the
model is not available offline and downloads are disabled, or the user turns the
feature off with ``E2A_SML_LLM_OFF=1``, ``llm_available()`` returns False and
every caller falls back to its deterministic heuristics.

Configuration (all optional):

* ``E2A_SML_LLM_MODEL`` — Hugging Face repo id or local directory; a
  comma-separated list is tried in order.  Defaults to ``Qwen/Qwen3-1.7B``
  then ``Qwen/Qwen3-0.6B``.
* ``E2A_SML_LLM_DEVICE`` — ``auto`` (default), ``cuda`` or ``cpu``.
* ``E2A_SML_LLM_DTYPE`` — ``auto`` (default), ``bfloat16``, ``float16``,
  ``float32``.
* ``E2A_SML_LLM_MAX_TOKENS`` — default answer budget (1024).
* ``E2A_SML_LLM_ALLOW_DOWNLOAD`` — ``0`` forbids fetching the model from the
  Hub; only an already-cached snapshot (or a local path) is used then.
* ``E2A_SML_LLM_OFF`` — ``1`` disables the local model completely.
"""

from __future__ import annotations

import json
import os
import re

# A small multilingual instruct model is the default: big enough to follow the
# JSON instructions, small enough to sit next to the NLP pipeline on one GPU.
DEFAULT_MODELS = ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B")
# Enough for a batch of emotion labels or a page of character JSON, small enough
# that a model ignoring the instruction cannot run away for minutes.
DEFAULT_MAX_TOKENS = 1024
# A listed model may still fail to load (bad download, out of memory): try a few.
_MAX_MODEL_ATTEMPTS = 3

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"^.*?</think>", re.DOTALL)


class LlmError(RuntimeError):
    """Raised when the local LLM cannot answer usefully."""


def extract_json(text: str) -> object:
    """Parse the first JSON value found in a model answer.

    Handles ```json fences and models that wrap JSON in prose or reasoning.
    """
    if not isinstance(text, str) or not text.strip():
        raise LlmError("the LLM returned an empty answer")

    match = _FENCE_RE.search(text)
    candidate = (match.group(1) if match else text).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(candidate):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[index:])
            return value
        except (json.JSONDecodeError, ValueError):
            continue

    raise LlmError("no JSON object found in the LLM answer")


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off", "")


def model_candidates() -> list[str]:
    """Model ids/paths to try, most preferred first."""
    configured = os.environ.get("E2A_SML_LLM_MODEL", "")
    listed = [item.strip() for item in configured.split(",") if item.strip()]
    return listed or list(DEFAULT_MODELS)


def allow_download() -> bool:
    """Whether the Hub may be contacted for a model that is not cached."""
    return _env_flag("E2A_SML_LLM_ALLOW_DOWNLOAD", True)


def _cache_dir_for(model: str) -> str:
    """The Hugging Face cache directory a repo id would live in."""
    from huggingface_hub.constants import HF_HUB_CACHE

    return os.path.join(HF_HUB_CACHE, "models--" + model.replace("/", "--"))


def _is_cached(model: str) -> bool:
    """True for a local directory or a repo id already in the HF cache."""
    if os.path.isdir(model):
        return True
    try:
        return os.path.isdir(_cache_dir_for(model))
    except Exception:  # huggingface_hub is a dependency, but stay safe
        return False


def _load_model(model_id: str, local_only: bool, dtype):
    """Load the tokenizer and model (kept in one place so tests can patch it)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, local_files_only=local_only
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, local_files_only=local_only
        )
    except TypeError:
        # Older transformers spell the argument ``torch_dtype``.
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, local_files_only=local_only
        )
    return tokenizer, model


class LocalLlmClient:
    """Lazily loaded in-process chat model."""

    def __init__(
        self,
        model: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        generator=None,
    ) -> None:
        self._configured_model = model or None
        self._device = device or os.environ.get("E2A_SML_LLM_DEVICE") or "auto"
        self._dtype = dtype or os.environ.get("E2A_SML_LLM_DTYPE") or "auto"
        # A callable ``(system, user, max_tokens) -> str`` used by the tests and
        # by anyone embedding the tool without a GPU.
        self._generator = generator
        self._model = None
        self._tokenizer = None
        self._model_id: str | None = None
        self._device_name: str | None = None

    # -- availability -------------------------------------------------------

    def candidates(self) -> list[str]:
        if self._configured_model:
            listed = [
                item.strip()
                for item in self._configured_model.split(",")
                if item.strip()
            ]
            return listed or [self._configured_model]
        return model_candidates()

    def available(self) -> bool:
        """True when a model could be loaded (cheap: nothing is loaded here)."""
        if self._generator is not None:
            return True
        if _env_flag("E2A_SML_LLM_OFF", False):
            return False
        try:
            import torch  # noqa: F401  (heavy, but needed to run a model)
            import transformers  # noqa: F401
        except Exception:
            return False
        if allow_download():
            return True
        return any(_is_cached(model) for model in self.candidates())

    @property
    def model(self) -> str:
        """The model id that is loaded (or the first candidate)."""
        return self._model_id or self.candidates()[0]

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # -- loading ------------------------------------------------------------

    def _resolve_dtype(self):
        import torch

        named = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if self._dtype in named:
            return named[self._dtype]
        if self._device_name == "cuda":
            return torch.bfloat16
        return torch.float32

    def _ensure_loaded(self, model_id: str) -> None:
        if self._model is not None:
            return
        import torch

        device = self._device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device_name = device
        dtype = self._resolve_dtype()

        tokenizer, model = _load_model(
            model_id, local_only=not allow_download(), dtype=dtype
        )
        model = model.to(device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._model_id = model_id

    # -- generation ---------------------------------------------------------

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        """Return the model's raw answer to a system+user prompt."""
        budget = int(
            max_tokens
            or os.environ.get("E2A_SML_LLM_MAX_TOKENS")
            or DEFAULT_MAX_TOKENS
        )

        if self._generator is not None:
            return self._generator(system, user, budget)

        last_error = "no model answered"
        for model_id in self.candidates()[:_MAX_MODEL_ATTEMPTS]:
            try:
                self._ensure_loaded(model_id)
                return self._generate(system, user, temperature, budget)
            except Exception as exc:
                last_error = f"{model_id}: {exc}"
                self.unload()
        raise LlmError(f"local LLM unavailable: {last_error}")

    def _generate(
        self, system: str, user: str, temperature: float, budget: int
    ) -> str:
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._apply_template(messages)
        encoded = self._tokenizer(prompt, return_tensors="pt")
        encoded = {
            key: value.to(self._device_name) for key, value in encoded.items()
        }
        input_len = encoded["input_ids"].shape[1]

        kwargs = {
            "max_new_tokens": budget,
            "pad_token_id": self._tokenizer.pad_token_id
            or self._tokenizer.eos_token_id,
            "repetition_penalty": 1.05,
        }
        if temperature and temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            # Greedy: drop the sampling flags entirely, otherwise the
            # tokenizer's default temperature/top_p/top_k trigger a warning.
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)

        with torch.inference_mode():
            output = self._model.generate(**encoded, **kwargs)
        answer = self._tokenizer.decode(
            output[0][input_len:], skip_special_tokens=True
        )
        return _THINK_RE.sub("", answer).strip()

    def _apply_template(self, messages: list) -> str:
        """Render the chat template with thinking disabled when supported."""
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def unload(self) -> None:
        """Drop the model and return its memory (VRAM included)."""
        self._model = None
        self._tokenizer = None
        try:
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # -- JSON ---------------------------------------------------------------

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> object:
        """Send a chat request and parse the answer as JSON."""
        return extract_json(self.chat(system, user, temperature, max_tokens))


_client: LocalLlmClient | None = None


def get_client() -> LocalLlmClient:
    """The process-wide client."""
    global _client
    if _client is None:
        _client = LocalLlmClient()
    return _client


def reset_client(client: LocalLlmClient | None = None) -> None:
    """Replace the process-wide client (tests, or after a config change)."""
    global _client
    _client = client


def llm_available() -> bool:
    """True when the local model can be used."""
    try:
        return get_client().available()
    except Exception:
        return False


def unload_model() -> None:
    """Free the loaded model (used once a book has been processed)."""
    if _client is not None:
        _client.unload()


# The client used to talk to an OpenAI-compatible HTTP server; keep the old
# name working for callers that construct it directly.
LlmClient = LocalLlmClient
