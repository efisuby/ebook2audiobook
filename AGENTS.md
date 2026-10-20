# AGENTS.md — ebook2audiobook (fork notes)

This checkout is a fork of
[DrewThomasson/ebook2audiobook](https://github.com/DrewThomasson/ebook2audiobook)
published at <https://github.com/efisuby/ebook2audiobook>. Everything from
upstream applies; this file only records the changes made in this fork so the
next agent does not have to rediscover them.

## Fork change: the `[emotion:LABEL]` SML tag

The companion project [E2A-SML](https://github.com/efisuby/E2A-SML) labels the
emotion of every spoken line and emits

```
[voice:…][emotion:LABEL]text[/emotion][/voice]
```

Upstream E2A does not know the tag, so `SML_TAG_PATTERN` (built from `TTS_SML`
keys) would not match it and it would be read aloud. The tag is therefore added
to the engine grammar.

### Files touched

| File | Change |
|------|--------|
| `lib/conf_models.py` | `TTS_SML["emotion"] = {"paired": True}` — the single source of truth for `sml_tag_keys`, `SML_TAG_PATTERN`, the translation-protection list and `normalize_sml_tags`. Also `TTS_SML_DEFAULT_EMOTION = "neutral"` and `TTS_SML_EMOTIONS` (per-label `temperature` / `top_p` / `speed` / `gain`). |
| `lib/classes/tts_engines/common/headers.py` | Export `TTS_SML_EMOTIONS`, `TTS_SML_DEFAULT_EMOTION` (so every engine sees them via `from ...common.headers import *`). |
| `lib/classes/tts_engines/common/utils.py` | `_convert_sml` gained an `elif tag == 'emotion':` branch: on open it snapshots `self.fine_tuned_params`, overrides only the keys already present, and stores `self.params["emotion_gain"]`; on close it restores both. A stack (`self._sml_emotion_stack`) makes nested tags safe. |
| `lib/classes/tts_engines/xtts.py` | In `convert()`, multiply the finished segment by `self.params.get("emotion_gain")` before `self.audio_segments.append(part_tensor)`. |
| `README.md`, `readme/README_rus.md` | Document `[emotion:LABEL]…[/emotion]` (labels, `neutral` no-op, gain on engines without tuned params). |

### Design notes for future edits

- **Adding any new SML tag means adding a `TTS_SML` entry first.** Everything
  else is derived from it; a tag missing from `TTS_SML` silently leaks into the
  spoken text.
- `TTS_SML_EMOTIONS` values are applied only to keys already present in
  `fine_tuned_params` (XTTS and Bark), so an engine without them gets a no-op —
  no crash.
- `neutral` is a no-op and is normally omitted by the writer.
- The `ipa` branch of `_convert_sml` is dead code — do not copy it.
- The companion E2A-SML project is the only writer of this tag; the SML file is
  still valid for stock E2A except for the unknown-tag behaviour above.

### How to check

E2A is normally run through Docker, so there is no ready local venv. Verify an
edit by parsing it and, when a subset of the deps is available, importing the
modules in isolation:

```bash
python -c "import ast; [ast.parse(open(p).read()) for p in ('lib/conf_models.py', 'lib/classes/tts_engines/common/utils.py', 'lib/classes/tts_engines/xtts.py')]"
python -c "from lib.conf_models import TTS_SML, TTS_SML_EMOTIONS; print(list(TTS_SML), sorted(TTS_SML_EMOTIONS))"
```

## Fork change: the Qwen3-TTS engine

Adds a `qwen3tts` engine (`lib/classes/tts_engines/qwen3tts.py`,
`class Qwen3TTS(TTSUtils, TTSRegistry, name='qwen3tts')`). It mirrors `xtts.py`
and uses the **Base** voice-clone models, so the per-character reference `.wav`
cast by E2A becomes the clone reference — x-vector-only, i.e. no transcript is
needed. Output is 24 kHz mono.

### Files touched

| File | Change |
|------|--------|
| `lib/conf_models.py` | `TTS_ENGINES["QWEN3TTS"] = "qwen3tts"` and a `default_engine_settings` entry (same 17-language map as XTTS, `samplerate: 24000`, `files: []`, `rating`). |
| `lib/classes/tts_engines/presets/qwen3tts_presets.py` | `models` with `"internal"` (`Qwen/Qwen3-TTS-12Hz-1.7B-Base`) and `"0.6B"` (`…-0.6B-Base`) presets, both `lang: "multi"`. |
| `lib/classes/tts_engines/qwen3tts.py` | New engine. `load_engine()` builds `Qwen3TTSModel.from_pretrained(..., device_map, dtype, cache_dir)`; `_voice_clone_prompt()` caches `create_voice_clone_prompt(ref_audio=…, x_vector_only_mode=True)` per voice; `convert()` splits on SML, honours `[voice:…]`/`[emotion:…]` and calls `generate_voice_clone(text, language, voice_clone_prompt)`. |
| `lib/classes/tts_engines/__init__.py` | `from .qwen3tts import Qwen3TTS` (this is what registers the engine). |
| `requirements.txt` | `qwen-tts` plus its runtime deps (`accelerate`, `librosa`, `einops`, `sox`). |
| `lib/classes/device_installer.py` | `no_deps_pkgs = ['qwen-tts']` (installed with `--no-deps`); `select_pkg('transformers')` returns the global pin `transformers==4.57.3` and `pip_overrides` forces `huggingface-hub<1.0` through a uv `--overrides` file. |
| `README.md`, `readme/README_rus.md` | `Qwen3-TTS` ticked in the engines list + a short usage note. |

### Design notes for future edits

- **The preset key `internal` is mandatory**: `default_fine_tuned = 'internal'`,
  `_update_gr_fine_tuned_list` falls back to it, and `_restore_interface` reads
  `models[default_fine_tuned]['files']`.
- No `qwen3tts_*` session keys were added (unlike `xtts_*`), so
  `self.fine_tuned_params = {}`. The `[emotion:LABEL]` tag therefore reaches
  this engine only through `self.params["emotion_gain"]` (applied in
  `convert()`), and the `[emotion:…]` branch in `_convert_sml` is a safe no-op
  for the tuned params.
- `qwen-tts` targets an old `transformers`; it is installed with `--no-deps` and
  its runtime dependencies must stay listed in `requirements.txt`. The runtime is
  held at `transformers==4.57.3` for it (see the next section), not at 5.x.
- No Gradio param tab is needed: the engine shows up in the dropdown through
  `get_compatible_tts_engines(language)`, derived from the `languages` map in
  `default_engine_settings`. `visible_xtts` / `visible_bark` stay `False` for it.
- `attn_implementation="flash_attention_2"` is deliberately **not** passed, to
  avoid a hard flash-attn dependency.
- `load_engine()` installs a logging filter that drops transformers' per-call
  "Setting `pad_token_id` to `eos_token_id`… for open-end generation" warning: the id
  it fills in lives in a nested Qwen3-TTS generation config and passing `pad_token_id`
  through `generate_voice_clone` does not reach it, so it would print once per
  sentence. It reports correct behaviour (pad = eos is what transformers picks).
  Only that record is dropped; other warnings from the same logger still print.
- **The engine's only spacing rules are the SML ones.** Qwen3-TTS pads every fragment
  it generates with leading and trailing silence (measured 200–500 ms per side), and a
  sentence is one fragment per `[voice:…]`/`[/voice]` run. Untouched, those paddings used
  to pile up with the tag silences and with the punctuation pause into seconds of dead
  air between two speakers. So `convert()`:
  - trims each generated fragment with `trim_audio()` (falling back to the raw audio if
    nothing survives the threshold, instead of failing the sentence),
  - appends the 0.3–0.6 s punctuation pause only when nothing else already pauses there —
    `_silence_follows()` looks past structural tags (`voice`, `emotion`) and past
    punctuation-only fragments to see whether a `[break]`/`[pause]` is coming — and
    always on the last fragment that carries words (`last_audible`), otherwise two
    sentences would be glued together now that the trailing padding is gone.
  Nothing else spaces the fragments out: in particular a `[voice:…]` switch inserts no
  pause of its own — spacing at a speaker change comes from the tags the SML writer put
  there.
- `unload_tts_manager()` needs no `engine.unload()`; it drops `loaded_tts[key]`
  and frees the GPU.

### How to check

```bash
python -c "import ast; [ast.parse(open(p).read()) for p in ('lib/conf_models.py', 'lib/classes/tts_engines/qwen3tts.py', 'lib/classes/tts_engines/presets/qwen3tts_presets.py', 'lib/classes/tts_engines/__init__.py', 'lib/classes/device_installer.py')]"
```

A live run needs `pip install --no-deps qwen-tts accelerate librosa einops sox`
and the `Qwen/Qwen3-TTS-12Hz-1.7B-Base` weights (~4.5 GB).

## Fork change: the Silero TTS engine

Adds a `silero` engine (`lib/classes/tts_engines/silero.py`,
`class Silero(TTSUtils, TTSRegistry, name='silero')`) for **Russian only**. Silero
cannot clone a voice: the per-character reference file cast by E2A only picks one of
the model's five built-in speakers. It runs on CPU (a sentence costs ~50–100 ms), so
the GPU is left to the other engines. Output is 24 kHz mono.

### Files touched

| File | Change |
|------|--------|
| `lib/conf_models.py` | `TTS_ENGINES["SILERO"] = "silero"` and a `default_engine_settings` entry: one language (`{"rus": "ru"}`), `samplerate: 24000`, `files: []`, `voices: {}`, `rating {"VRAM": 0, "CPU": 5, "RAM": 2, "Realism": 4}`. |
| `lib/classes/tts_engines/presets/silero_presets.py` | `models` = `internal` (`v5_6_ru`, the default), `v5_6_ru` (the same checkpoint under its own name, so the GUI list shows the version), `v5_5_ru` and `v4_ru`; `repo` is the `.pt` URL on `models.silero.ai`, each entry carries the optional `"stress"` flag. |
| `lib/classes/tts_engines/silero.py` | New engine. `load_engine()` downloads the `.pt` with `torch.hub.download_url_to_file` and unpacks it with `torch.package.PackageImporter(...).load_pickle('tts_models', 'model')`; `_speaker_for()` maps a voice file to a built-in speaker; `convert()` copies `qwen3tts.convert()`. |
| `lib/classes/tts_engines/__init__.py` | `from .silero import Silero` (this is what registers the engine). |
| `requirements.txt` | `silero-stress` — the text pre-processing (stress marks + homograph resolution). |
| `lib/classes/device_installer.py` | `no_deps_pkgs += ['silero-stress']`, so uv does not resolve its `torch` from PyPI. |
| `README.md`, `readme/README_rus.md` | `Silero` in the engines list + a usage note (licence, download, CPU, stress). |

The checkpoint is a torch-package file loaded straight from disk, so the synthesis itself
needs nothing but `torch`; only the stress pre-processing is an extra package. The `silero` PyPI package is deliberately
not used — it only wraps the same download.

### Design notes for future edits

- **Only Russian.** The settings carry a single `{"rus": "ru"}` and every preset points at
  a `…/tts/ru/*.pt` URL. Silero's other families (CIS/Turkic/Caucasian, the jit `*_8khz`/
  `*_16khz` models) live at other URLs and some need the `torch.jit` path instead of
  `torch.package` — that is where to extend.
- **The `lang` field of a preset must be `"multi"` or an E2A language code** (`"rus"`, like
  the `"eng"` entries of `xtts_presets.py`), *not* the ISO1 code: the GUI builds the model
  list with `details.get("lang") in ("multi", session['language'])`
  (`_update_gr_fine_tuned_list` in `lib/gradio.py`), so an unmatched value leaves the
  dropdown empty and gradio then refuses the default with
  `Value: internal is not in the list of choices: []`.
- **`v5_6_ru` is the best Russian model** (auto-stress, homographs, questions, emphasis) and
  is what `internal` points at; the same checkpoint is also listed as `v5_6_ru`, because the
  GUI dropdown only shows the preset keys and `internal` does not tell the user which
  version it is. `v5_5_ru` is the same voices without the emphasis
  intonation, and `v4_ru` is the smaller previous generation (~40 MB, the only one that also
  advertises `speaker='random'`). The v5 family has exactly five speakers — `aidar`, `baya`,
  `kseniya`, `eugene`, `xenia` — and does **not** accept `speaker='random'`.
- **Voice mapping** (`_speaker_for`): a voice file whose stem is a speaker name is used
  as-is; otherwise the gender is read from the directory (`…/adult/male/…` vs
  `…/adult/female/…`, female checked first, deterministic md5 fallback when neither is
  there) and the speaker is handed out round-robin from `speaker_pools` (male
  `aidar`/`eugene`, female `baya`/`kseniya`/`xenia`), cached in `params['voice_speakers']`
  so a narrator keeps one speaker for the whole book. The pools are intersected with the
  speakers the loaded checkpoint really has (read from `engine.speakers`), so a variant that
  ships fewer voices still maps.
- **Stress pre-processing** (`_accent`): every plain text fragment is run through
  `silero_stress.load_accentor()` before `apply_tts()`. The accentor is loaded lazily on the
  first fragment (~1.5 s once per engine) and costs ~0.02 s per sentence. The v5/v4 archives
  already embed an accentor and a homograph solver (`models/accentor.py`,
  `models/homosolver.py` inside the package), but they honour explicit `+` marks —
  `зам+ок` and `замо+к` synthesize differently — so the newer standalone model decides.
  Measured on `v5_6_ru`, the pre-accented text gave byte-identical audio to the built-in
  accentor on most sentences and a different take on `Уже 3 часа мы едем в поезде.`.
  Only the text between SML tags is accented (tags and voice paths must never be touched),
  so a homograph is solved per fragment, without the full sentence around it.
  `Silero.stress_enabled = True` is the class default, a preset entry overrides it with
  `"stress": False`, and a missing or broken `silero_stress` only prints a warning and falls
  back to the raw text — it never fails the sentence. The installed package (`silero-stress`
  1.5, wheel ~37 MB, 65 MB unpacked) ships the weights; nothing is downloaded at runtime.
- Spacing and `[emotion:…]` behaviour are copied from `qwen3tts`: `convert()` uses the
  same `trim_audio()` / `_silence_follows()` / `last_audible` rules (so no pause comes
  from a `[voice:…]` switch), and `fine_tuned_params = {}` means the emotion tag reaches
  this engine only as `params['emotion_gain']`.
- `samplerate: 24000`, not the model's 48 kHz: at 48 kHz the output peaks above 1.0
  (measured 1.043) while the final ffmpeg pass writes `-sample_fmt s16`, so it would
  clip. All three rates produce the same duration (it re-renders, not upsamples).
- The `.pt` source contains `'\^'`, so python emits
  `SyntaxWarning: invalid escape sequence` while compiling it; `load_engine()` swallows
  exactly that category.
- Not in `tts_engines_with_inner_speaker`, `tts_engines_from_coqui` or
  `tts_engines_with_custom_model`: the GUI keeps the XTTS-like voice flow (voices from
  files), because the speaker choice happens inside the engine.
- `apply_tts()` exposes no `speed` and no `ssml` switch (SSML goes through `ssml_text=`),
  but it does take `voice_path=`, which is the extension point if real cloning is wanted.
- **Licence:** the Silero weights are CC BY-NC-SA 4.0 (non-commercial); the Silero package
  code and `silero-stress` are MIT.

### How to check

```bash
python -c "import ast; [ast.parse(open(p).read()) for p in ('lib/conf_models.py', 'lib/classes/tts_engines/silero.py', 'lib/classes/tts_engines/presets/silero_presets.py', 'lib/classes/tts_engines/__init__.py')]"
```

A live run downloads the selected checkpoint from `models.silero.ai` into
`models/tts/silero/` (~150 MB for the `v5_*` ones, ~40 MB for `v4_ru`); `internal`,
`v5_5_ru` and `v4_ru` all synthesize, and both the engine smoke test and a headless
`--tts_engine silero --device cpu` run pass on torch 2.14.1+cu130.

The stress pre-processing needs `silero-stress` (installed with `--no-deps`, like the
rest of `requirements.txt`): `engine._accent('замок')` must return `'зам+ок'`, and with
`engine.use_stress = False` the same call must return the text unchanged. An A/B of
`convert()` with and without the marks must differ for a sentence whose reading the
marks change (e.g. `Уже 3 часа мы едем в поезде.`).

## Fork change: transformers pinned to 4.57.3

`qwen-tts` does not run on transformers 5.x: 5.x turned `@check_model_inputs()` into
`check_model_inputs(func)`, and removed the `PretrainedConfig` defaults the qwen
configs read (`pad_token_id`). Shimming the decorator only moves the failure one
step further, so the whole runtime is pinned instead.

| Pin | Where | Why |
|-----|-------|-----|
| `transformers==4.57.3` | `device_installer.select_pkg('transformers')` (it is in `device_pkgs`, so the pin rides along on every install call) | the only line `qwen-tts` supports |
| `tokenizers==0.22.2` | resolved automatically as a dependency | 4.57.3 aborts at import above `tokenizers==0.23.0`; the 0.23.0 release does not exist (0.23.0rc0, then 0.23.1…), so 0.22.2 is the newest allowed |
| `huggingface-hub<1.0` (0.36.2) | `device_installer.pip_overrides`, written to a uv `--overrides` file in `pip_cache_dir` | 4.57.3 declares `hub<1.0` (it imports `HTTPError` from `huggingface_hub.utils` and expects hub errors to subclass `requests.HTTPError` — both gone in hub 1.x/2.x), while `gradio==6.29.1` declares `hub>=1.16.0`. The resolver cannot satisfy both, so the override forces the older hub. gradio 6.29.1 imports, starts and runs on 0.36.2 (verified); E2A itself only uses hub via `hf_hub_download` / `snapshot_download` / `get_hf_file_metadata` / `hf_hub_url`, all present in both lines. |

Consequences to keep in mind:

- The transformers 5.x patches in `components/sitecustomize.py` are inert on this pin
  but must stay: they repair torch<2.6 `check_torch_load_is_safe`, the coqui-tts
  `isin_mps_friendly` import and the GPT2InferenceModel all-1s attention_mask.
- `--overrides` is only added to `uv pip install` (see `_uv_pip`), never to `uninstall`.
- Verified end to end on this stack (torch 2.14.1+cu130, CUDA): a headless `qwen3tts`
  run, a headless `xtts` run, and the qwen3tts engine smoke test (5 cases, with the
  `[emotion:…]` gain applied).
- After any transformers or gradio bump re-check both engines: transformers must stay
  below 5 unless `qwen-tts` ships a 5.x-capable release, and gradio must stay on a
  version that runs with `huggingface-hub<1.0`.

## Fork change: the SML pipeline (`booknlp/` + `sml_extractor/`)

The E2A-SML project is merged into E2A: `booknlp/` (vendored BookNLP plus the fork's
Russian pipeline) and `sml_extractor/` (BookNLP runner, character extraction,
descriptions, voice casting, emotions, SML writer) sit in the repository root and are
imported directly, so `app.py` and the GUI can run the whole book-to-SML path in the
same process. Runtime data is reused from the E2A-SML checkout: `data/` is a symlink to
`../E2A-SML/data` (7.8 GB — BookNLP models, the Qwen3-1.7B cache, the voice library) and
is gitignored; `SML_DATA_DIR` overrides the location.

### Files touched

| File | Change |
|------|--------|
| `booknlp/` | vendored verbatim (incl. `english/data/` resources, the fork's `russian/` pipeline, `LICENSE`) |
| `sml_extractor/` | ported verbatim minus `docker_entrypoint.py` |
| `sml_extractor/character_profile.py` | new: per-character descriptions |
| `tests/` | the 7 E2A-SML test modules (without `test_web_gui.py`) + `test_character_profile.py` |
| environment | `sentence-transformers`, `ru_core_news_sm`, `en_core_web_sm`, `pymorphy3`, `pymorphy3-dicts-ru`, `dawg-python`, all installed with `--no-deps` (the spaCy models from the GitHub release wheels) |

### Design notes for future edits

- Character dicts keep the E2A-SML shape: `normalized_name`, `inferred_gender`,
  `inferred_age_category`, `tts_engine`, `language`, `voice` (+ optional `description`,
  `description_source`, `key_character`). `extract_characters()` returns
  `booknlp_data['characters_simple']['characters']` untouched when BookNLP wrote it; the
  Russian gender/age inference in `core.py` (`_infer_gender_from_russian_name`, the
  `_RU_*` tables) only runs on the fallback path.
- `character_profile.describe_characters()` writes `description` and
  `description_source` ∈ `user`/`llm`/`heuristic`: text the user supplied is never
  overwritten, the heuristic is always written first, and the model only replaces it for
  characters it actually describes.
- Evidence is deliberately narrow: quotes come from the dialogue index, mentions are
  *the narration sentence containing a name variant* (`_sentences`, not a ±200 char
  window — a window leaks the next character's material into the description). Variants
  are searched with a leading `\b` but no trailing one, so Russian case endings match.
- Characters without a line of their own and with fewer than `_MIN_MENTIONS` (2)
  mentions are not sent to the model at all, and neither is `Narrator`: on that little
  material the model invents. The heuristic stays
  (`взрослый мужчина, 3 реплик. Характерные слова: …` / `Голос рассказчика, …`).
- The prompt forbids inventing a biography, turning the age category into a number or
  inventing looks, and asks for flowing prose: the first version mimicked the rule
  labels and answered in a `«Возрастная категория: … Внешность: …»` template.
- Batch size from `E2A_SML_PROFILE_BATCH` (default 8); the model is the shared
  `llm_client` (Qwen3-1.7B, `E2A_SML_LLM_*` env), and any `LlmError`/exception only
  prints and leaves the heuristics in place — descriptions never fail a book.

### How to check

`env -u LD_LIBRARY_PATH python_env/bin/python -m unittest discover -s tests` → 108 tests
OK (85 ported + 23 profile). For the real path run `/tmp/opencode/profile_e2e.py`:
BookNLP (small, `rus`) over a short Russian text, then `describe_characters` with the
real model — `Narrator` and thin characters must stay `heuristic`, the rest `llm`, and
no description may mention another character's features.

## Fork change: the SML stress pre-processing (`sml_extractor/stress.py`)

### Files touched

| File | Change |
| --- | --- |
| `sml_extractor/stress.py` | New: accents the plain prose of an SML text with silero-stress (`замок` → `зам+ок`, mark = `+` before the vowel — the form the checkpoints were trained on), or strips those marks again. |
| `tests/test_stress.py` | New: 13 tests (accenting only outside tags, non-`+` plus signs survive, engine routing). |
| `lib/classes/tts_engines/silero.py` | `_accent()` translates the leftover U+00B4 (`´`) mark → `+` before `apply_tts()`; its own lazy accentor stays behind the preset's `"stress"` flag. |

### Design notes for future edits

- Only Silero's checkpoints understand the marks: every other engine would read a mark
  as a character, so `stress_for_engine(text, tts_engine)` accents the text for `silero`
  and strips the marks for everything else.
- **Stress is the LAST filtering stage.** The marks are written only after every
  normalizer in `lib.core.filter_blocks` — `math2words()` would otherwise verbalize a
  `+` into the word «плюс» (its `language_math_phonemes['rus']['+']`). Incoming marks
  are stripped before `non_text_filter()` and re-applied after `restore_sml()`, gated by
  `session.get('sml_use_stress', True)` and the legend `msg_apply_stress`.
- The mark kept in the SML document **and sent to the model is `+`** (`Г+ород`) — the
  notation silero-stress returns and the Silero checkpoints were trained on. An earlier
  variant of this file wrote U+00B4 (`´`) instead; that mark is never written again:
  `accent_text()` translates it back to `+`, `strip_text()` removes `´` before a vowel,
  `+` before a vowel and combining U+0301 after a vowel. **Do not reintroduce `´`** —
  the request is that the model receives `+` text.
- A mark is only a symbol *before a vowel*, so `2+2`, `A+B` and `C+1` are never changed.
- SML tags and the voice paths inside them are never touched: `_outside_tags()` splits on
  a local copy of the tag regex and only decorates the text chunks. Keep the tag names in
  sync with `TTS_SML` in `lib/conf_models.py` — the module is deliberately independent of
  the rest of E2A so that it can be tested and reused on its own.
- The local tag regex also matches the closing tags (`[/voice]`), so an accented block
  keeps its exact shape; `strip_sml` of an accented text returns exactly the input, which
  makes the non-Silero route safe when a book already carries marks.
- `get_accentor()` loads the model lazily (one ~1.5 s hit per process) and remembers a
  failure (`[stress] silero-stress unavailable, the text is left unaccented: …`), so a
  missing package degrades to the unaccented text instead of failing the book. Use
  `reset_accentor()` in tests.

### How to check

`env -u LD_LIBRARY_PATH python_env/bin/python -m unittest discover -s tests` → 121 tests
OK (85 ported + 23 profile + 13 stress). With the real package installed,
`accent_sml('[voice:/v/rus/adult/male/Petr.wav]Замок был на двери, а замок стоял на реке.[/voice]')`
must mark the prose with `+` and leave the tag and its path verbatim, `strip_sml` of the
result must return the input, and `stress_for_engine(text, 'xtts')` must strip the marks.

## Fork change: the SML wizard glue (`sml_extractor/pipeline.py`)

The step-by-step path the wizard walks lives here, so the GUI only has to call
one function per step and keep the paths in the session: `analyze_book` (BookNLP
+ characters), `describe_characters`, `cast_characters`, `character_rows`,
`sample_line`, `preview_lines`, `build_sml`, `sml_blocks`, `stress_preview`.

### Files touched

| File | Change |
| --- | --- |
| `sml_extractor/pipeline.py` | New: the glue between the ported pipeline and E2A. |
| `sml_extractor/sml_generator.py` | `book_segments()` and `render_sml_text()` are public now; `generate_sml_output()` only writes what `render_sml_text()` returns, so the document has one builder. |
| `tests/test_pipeline.py` | New: 27 tests over fake book data, a temporary voice library and patched accentor/emotion steps. |

### Design notes for future edits

- The module imports no E2A code: language, engine and the library root are arguments,
  which is what lets the tests drive the whole path. `APP_ROOT` is the E2A checkout
  (where `voices/` and `data/` live) so a caller can pass nothing and still get the right
  defaults.
- `analyze_book()` keeps the token and quote tables in memory and returns paths only:
  the session stores paths and `load_book()` re-reads the analysis, so a restart costs a
  load, not a BookNLP run.
- `cast_characters()` returns the cast *and appends* the records to the `characters` list
  it was given: the narrator is always part of it, and a speaker the analysis missed gets
  an `unknown`/`adult` record, so a later `describe_characters()` covers that speaker too.
  Voice precedence: the override map, then the character's own `voice`, then the
  automatic assignment by gender and age. Library paths come back relative
  (`voices/<lang>/<age>/<gender>/<file>`) and are made absolute against the root — that
  relative form is what `[voice:…]` tags in a saved document should carry.
- `character_rows()` sorts the narrator first, then by the number of quotes, then by the
  order of `characters` (not alphabetically) so the table matches the book.
- `sample_line()` never raises on a book without data: the wizard asks for a sample before
  a character is known to have a line.
- `build_sml()` runs the emotion model (which itself falls back to the heuristic), renders
  the document and only then applies the stress step: marks for Silero, stripped for every
  other engine, so one document can be re-targeted. `sml_blocks()` is the bridge to E2A —
  the writer puts every voice block on one line, and one line is one block.
- `preview_lines()` is the "one line per character plus the opening of the book" text the
  user reads before anything is synthesized; it is text, not audio.

### How to check

`env -u LD_LIBRARY_PATH python_env/bin/python -m unittest discover -s tests` → 148 tests
OK (85 ported + 23 profile + 13 stress + 27 pipeline).
`/tmp/opencode/pipeline_smoke.py` walks the whole path without BookNLP or the model: a
four-line book with two quotes, then cast, rows, sample lines, preview, stress preview
(`+` for `silero`, clean for `xtts`) and the SML blocks.

## Fork change: the book-to-SML wizard (`lib/sml_wizard.py` + the GUI)

The interface drives the whole path in one place: **«Анализ книги» → cast/describe →
«Собрать SML» → listen to one line per character → «Озвучить»**, with a preview and a
cancel button before the book is really synthesized. `lib/sml_wizard.py` holds the
session-driven logic (no Gradio), `lib/gradio.py` only calls one function per button.

### Files touched

| File | Change |
| --- | --- |
| `lib/sml_wizard.py` | New: `analyze`, `rows`, `names`, `voice_choices`, `set_voice`, `voice_of`, `sample`, `preview`, `build`, `apply`, `cancel`, `save`, `restore`, `state`, plus the cast editing (`remove_character`/`removed`, `set_gender`/`set_age`, `set_selected`), the Silero built-in voices (`silero_choices`, `builtin_speaker`, `_write_silence`), the parsed-book cache and the source digest. |
| `lib/gradio.py` | The `gr_sml_*` component group (after `gr_convert_btn`) and the `_click_gr_sml_*` handlers + event wiring (before `########### Main chains`), plus the `_restore_sml_wizard` step in the session-restore chain. |
| `lib/core.py` | `sml_output_dir`, `sml_book_id`, `sml_characters`, `sml_removed`, `sml_selected`, `sml_text`, `sml_applied`, `sml_multi_voices` added to `set_session` (so they survive a reload) and **removed** from `reset_ebook_session`. |
| `sml_extractor/pipeline.py` | `sample_text` (`DEFAULT_SAMPLE_WORDS = 40`, `MAX_SAMPLE_WORDS = 90`) — a multi-sentence listen fragment. |
| `tests/test_sml_wizard.py` | New: 57 tests over a fake session, patched pipeline steps and a temporary voice library. |

### Design notes for future edits

- `_start_conversion` calls `reset_ebook_session(session_id, force=True,
  filter_keys=False)` at the start of **every** conversion. That is why the `sml_*` keys
  live only in `set_session` and not in `reset_ebook_session` — otherwise «Отмена» could
  not return to the voice choice, and the SML document would be wiped by the very
  conversion that uses it.
- `lib/sml_wizard` is imported lazily (`_sml_wizard()` inside each handler) so a broken
  ML setup cannot stop the app from starting. Its labels are literal Russian strings and
  are deliberately **not** added to the `legends` translation tables.
- The wizard components are deliberately not part of `outputs_refresh_interface`,
  `outputs_disable_components` or `outputs_enable_components`: those lists are
  index-based and fragile. Consequence: the panel stays interactive while a conversion
  runs, and `_refresh_interface` does not touch it.
- «Озвучить» does not start a conversion of its own: it writes the SML into
  `gr_ebook_textarea`, switches `gr_ebook_mode` to `TEXT`, and the normal chain
  (`_chain_check_override` → `_check_override_ebook` → `gr_event.change` →
  `_start_conversion`) takes over. `_check_override_ebook` stores
  `session['ebook_textarea']` and bumps `gr_event`, so the wizard reuses the whole TEXT
  path unchanged.
- «Отмена» handles both states: while converting it sets `cancellation_requested` (the
  same flag the per-sentence loop polls), otherwise it clears `sml_applied` and returns
  to the picker. «Стоп» is the explicit interrupt for a running synthesis.
- `analyze()` skips the BookNLP run when the md5 of the source file is unchanged
  (digest kept in a module dict **and** in `characters.json`), so pressing «Анализ книги»
  twice — or re-entering the panel — is cheap.
- `book_data()` returns the same shape on a cache hit and after a reload: the tables
  themselves plus `output_dir`/`book_id` (never the `load_book()` wrapper). `rows()`
  falls back to the session records when the tables are gone.
- The listen preview is written to `<session_dir>/sml/__sml_preview.<fmt>` (not
  `process_dir`, which may not exist before the first conversion), with the engine,
  `fine_tuned` and `model_cache` backed up and restored around it.
- Voice choices come from the `sml_extractor` scanner (`scan_voice_library`, accepts
  `.mp3`), not from E2A's own `voices/<lang>` dropdown, which only lists `.wav` and is
  nearly empty for `rus`. The dropdown values are absolute paths.
- **Deleting a character** (`remove_character`) drops the record from
  `sml_characters` and remembers the name in `sml_removed`. The name is *not* a mute
  channel: `_voice_map` points it at the **narrator's** voice, so the lines it was
  credited with are read by the narrator instead of being left to the block voice.
- **Silero built-in voices** (`aidar`, `eugene`, `baya`, `kseniya`, `xenia`) are offered
  as extra dropdown entries labelled `[silero] <speaker> · <gender>`. There is no engine
  change: the wizard writes a 0.05 s silent WAV per speaker under
  `<session_dir>/sml/voices/<lang>/adult/<gender>/<speaker>.wav`, which satisfies
  `_convert_sml`'s existence check, gives `_set_voice` a path carrying the language (so
  it is passed straight through) and `_voice_gender` the right gender, and whose stem
  lets `Silero._speaker_for` pick the real built-in. `_voice_map` removes those marker
  paths for any engine other than `silero`, so a document built for XTTS/Qwen can never
  contain them. The marker lives only in the dropdown **label**; the value stays the path.
- **Gender and age** are edited on the character record (`inferred_gender` /
  `inferred_age_category`) and immediately visible in the table (`character_rows` reads
  those two fields). `_set_field` returns early when the value is unchanged, because the
  restore step re-renders the dropdowns and must not invalidate a document the user
  already built; a real change drops `sml_text` (and thus forces «Собрать SML» again).
  The two dropdowns are wired with `.input` (user-only) rather than `.change` for the
  same reason.
- **Session persistence**: every `sml_*` key is an ordinary session key, so the existing
  save timer already stores it in localStorage and `restore_session_from_data` brings it
  back. What was missing was the panel: `_restore_sml_wizard(session)` (a step in the
  `gr_restore_session.change` chain, right before the glassmask is hidden) rebuilds the
  table, the pickers, the description, the gender/age dropdowns and the preview from
  `wizard.state(session)`, which also calls `wizard.restore(session)` as a disk fallback.
  Consequence: after a reload/restart the user lands back on the same stage.

### How to check

`env -u LD_LIBRARY_PATH python_env/bin/python -m unittest discover -s tests` → 210 tests
OK (85 ported + 23 profile + 13 stress + 27 pipeline + 5 silero + 57 wizard).

`env -u LD_LIBRARY_PATH python_env/bin/python -c "from lib.gradio import build_interface; build_interface({'script_mode': False, 'is_gui_process': True, 'share': False})"`
must build the Blocks without error (it does, Gradio 6.29.1).

In the running app: «Анализ книги» fills the table and the descriptions, picking a
character shows its description, voice, gender and age, «◉» synthesizes a longer fragment,
«Собрать SML» shows the text preview, «Озвучить» starts the normal conversion,
«Отмена»/«Стоп» interrupt it. «🗑 Удалить персонажа» removes a bogus name (the narrator
reads its lines); the gender/age dropdowns edit the cast; with the Silero engine the voice
list also offers the `[silero]` built-ins; «Разные голоса» off reads the whole book with the
narrator's voice. Reloading the page keeps the stage.

## Fork change: the «Сброс сессии» button

`gr_session_reset_btn` (the session group, next to `gr_session_switch_btn`) drops the
work of the current session — the loaded book, its finished audio and the SML wizard —
while keeping the settings (engine, model, voice, language, output format).

### Files touched

| File | Change |
| --- | --- |
| `lib/gradio.py` | `gr_session_reset_btn`, `_click_gr_session_reset_btn()` and its `.click()` wiring (single output: `gr_restore_session`). |

### Design notes for future edits

- The handler runs `reset_ebook_session(force=True, filter_keys=False)` and then clears
  the source keys (`ebook_src`, `ebook_list`, `ebook_selected`, `ebook_textarea`,
  `audiobook`) **and every `sml_*` key** by hand: `reset_ebook_session` deliberately
  leaves `sml_*` alone (the wizard needs them to survive a conversion and «Отмена»), so
  a real reset has to drop them itself.
- The single output is `gr_restore_session`: handing the cleared session back as JSON
  runs the ordinary restore chain (`_change_gr_restore_session` → `_restore_interface` →
  `_restore_ui_language` → `_restore_audiobook_player` → `_restore_sml_wizard` →
  `init_interface()`), the same path a page load takes. That is why the handler lists no
  components of its own — everything the chain renders (file slot, textarea, player,
  lists, wizard) is refreshed from the cleared session, and the save timer then stores
  that state in localStorage so a reload does not bring the book back.
- The payload must carry `status = ready`: `_change_gr_restore_session` only forces its
  values in when `data['status'] == status_tags['READY']`.
- While `status` is `converting` or `edit` the click only shows a warning — the session
  object is still owned by the running conversion.
- The payload is `{k: v for k, v in session.items() if k not in save_session_keys_except}`,
  exactly what `_update_gr_save_session()` stores.

## Fork change: the «Разные голоса» switch

`gr_sml_multi_voice` (a checkbox right under «Анализ книги» / «Собрать SML»)
switches the voicing between the cast and one voice for the whole book. **On is
the default** and is the old behaviour: every character is read with its own
voice. **Off** collapses every voice onto a single one — the narrator's — so the
whole book is read by one voice.

### Files touched

| File | Change |
| --- | --- |
| `lib/sml_wizard.py` | `multi_voices`, `set_multi_voices`, `unified_voice`, `effective_voice`, `effective_voices`, `voice_picker_enabled`; `rows`, `preview`, `build` and `_voice_map` now go through `effective_voices`/`effective_voice`, and `state` returns `multi`. `save`/`restore` carry `multi_voices`. `analyze` skips casting + descriptions when off, and reuses the BookNLP output on a re-run. |
| `sml_extractor/pipeline.py` | `preview_lines()` passes `voices=voices` to `character_rows()`, so the preview line shows the voice that will really be used. |
| `lib/gradio.py` | The `gr_sml_multi_voice` checkbox and the `_click_gr_sml_multi_voice` handler + its `.input()` wiring; the character-pick/table/voice-pick/gender/age handlers and `_restore_sml_wizard` render `effective_voice` and set the dropdown's `interactive` from `voice_picker_enabled`; `_click_gr_sml_listen_btn` synthesizes with `effective_voice` (off ⇒ the narrator's voice); turning the switch on with an uncalled cast asks for «Анализ книги». |
| `lib/core.py` | `sml_multi_voices` (default `True`) added to `set_session`. |
| `tests/test_sml_wizard.py` | `MultiVoiceTests` (12 tests) and `SingleVoiceAnalysisTests` (4 tests). |

### Design notes for future edits

- **The cast is never rewritten.** Off only changes what `_voice_map`/`rows`/
  `preview`/the dropdown render; turning the switch back on restores every
  per-character voice, so the user never loses the casting work.
- **Off skips the two slow steps of `analyze`** — `cast_characters` (voice
  selection) and `describe_characters` (the LLM descriptions) — because with one
  voice the cast is never used. BookNLP itself still runs: it is what splits
  narration from dialogue, and only dialogue carries the `[emotion:…]` labels,
  which are kept in one-voice mode. The characters are still produced (and the
  table still shows them), they simply have no `voice`/`description`.
- The «already analysed» early return only short-circuits when the switch is on
  **or** the cast already has voices (`cast_ready`), so flipping the switch on
  after a one-voice analysis really runs the casting.
- The re-run reuses the BookNLP output already on disk (`pipeline.load_book`)
  instead of re-running the model, so switching the mode costs only the casting
  and the descriptions, not the parse.
- `unified_voice()` = the **narrator's** voice, falling back to the first voice in
  the cast when there is no narrator. With no voice at all, off degrades to the
  unchanged map (nothing is collapsed).
- **`sml_multi_voices` is deliberately NOT in `reset_ebook_session`** — like the
  other `sml_*` keys, it has to survive every conversion (including «Озвучить»)
  so that off stays off while the book is being read. The real reset is
  `gr_session_reset_btn`, which drops the whole `sml_*` block by hand; add it
  there if a new key is introduced.
- `_voice_map` order matters: `effective_voices()` first, then the deleted-name →
  narrator override, then the Silero-marker filter. Collapsing before the
  narrator override is safe because both end up on the narrator's voice.
- With off, the **voice dropdown stays editable only on the narrator's row**
  (`voice_picker_enabled`); that is where the single voice is picked. The other
  rows render the narrator's voice but their edits would be ignored. The switch
  itself is wired with `.input()` (user-only), like the gender/age dropdowns, so
  a programmatic re-render cannot toggle the mode behind the user's back.
- The switch is **not** in `outputs_refresh_interface`/`outputs_disable_components`
  (index-based lists) — same choice as the rest of the wizard panel.
- Naming: the checkbox is «Разные голоса», the session key is `sml_multi_voices`.

### How to check

`env -u LD_LIBRARY_PATH python_env/bin/python -m unittest discover -s tests` → 237 tests
OK (233 before the one-voice analysis skip, +4). `build_interface(...)` still builds
the Blocks.

In the running app: analyse a book, uncheck «Разные голоса» — the table and the
preview show the narrator's voice for every character and «Собрать SML» reports
`Один голос для всех: <voice>`; check it again and every cast voice is back.
While a document is already built, toggling asks for «Собрать SML» again.
Analysing with the switch off skips the descriptions and the casting (watch for
`[sml] Режим одного голоса: …` in the console); flipping it on afterwards and
pressing «Анализ книги» runs just those two steps (the parse is reused).

## Fork change: Silero survives unpronounceable fragments

### Files touched

| File | Change |
| --- | --- |
| `lib/classes/tts_engines/silero.py` | `convert()` catches the checkpoint's `ValueError` per fragment and skips it, and writes a short silence when a whole sentence produced no audio. |
| `tests/test_silero_convert.py` | New: 5 tests over a fake checkpoint (`Silero.__new__`, faked `apply_tts`/`audio_save`), so no model is loaded. |

### Design notes for future edits

- The v5 checkpoints raise a **bare `ValueError` with an empty message** from
  `process_simple_text` for any fragment they cannot pronounce: digits, latin words,
  stray markup. Measured on `v5_6_ru` (probe `/tmp/opencode/silero_probe.py`):
  `'1'`, `'12'`, `'II'`, `'OK'`, `'Hello'`, `'H2O'`, `'Mr. Smith…'`, `'№5'`, `'[1]'`,
  `'@user'`, `'—'`, `'...'`, `'***'` fail; `'5-6'`, `'-'`, `'--'`, `'«-»'` and anything
  containing a Cyrillic letter succeed. Fragments with no alphanumeric character never
  reach `apply_tts` (the loop skips them first), so in practice it is **digits and
  latin text** that trigger the error.
- One `return False` from `convert()` aborts the entire book
  (`convert_chapters2audio` stops on `run=False`), and a sentence with no output file
  later trips `combine_audio_sentences` (`error_missing_sentence_files`). That is why
  the fix does two things: skip the fragment **and** guarantee a file. A sentence that
  yields no audio gets `0.3 s` of silence (`torch.zeros(1, samplerate*0.3)`), keeping
  the per-sentence index alive.
- The `except ValueError` sits between `except IndexError` and the generic `except
  Exception`, so a real synthesis failure still returns `False` and, as before, lets
  `core.py` unload the engine.
- Consequence to keep in mind: a latin-only or digit-only line is now **silent**, not
  read aloud. If that ever matters, the place to add number/transliteration handling is
  `Silero.convert()` right before `apply_tts`, not the checkpoint.

### How to check

`env -u LD_LIBRARY_PATH python_env/bin/python -m unittest discover -s tests` → 210 tests
OK (85 ported + 23 profile + 13 stress + 27 pipeline + 5 silero + 57 wizard).

Real-checkpoint run (`/tmp/opencode/silero_real_check.py`) with `v5_6_ru`:
`'12345'`, `'...'` and `'Mr. Smith'` each return `(True, None)` with a 57680-byte silent
wav instead of raising, while `'Привет, мир!'` and `'Глава 2. Он пришёл.'` still
synthesize normally.

(The suite count above predates the «Разные голоса» switch (+12 wizard tests), the
sentence-pause fix (+5 silero tests), the console logs, the 1024 sentence cap
(+6 split tests) and the one-voice analysis skip (+4 wizard tests) — the current run
is 237 tests OK.)

## Fork change: the sentence pause is recovered from the terminator

The Silero engine used to pause after every fragment that did not end on a word
character. Because the pipeline splits a block **at** its punctuation, a fragment
cut at a comma («…дела,») looked like a sentence end and got a full 0.3–0.6 s pause:
the narrator seemed to stop at every comma.

### Files touched

| File | Change |
| --- | --- |
| `lib/classes/tts_engines/silero.py` | `_terminator_after()` restores the pause decision from the untrimmed sentence; `_raw_source()` finds the fragment inside it; `_convert_parts` keeps the split parts of the current sentence. |
| `tests/test_silero_convert.py` | New `TerminatorPauseTests` (5 tests). |

### Design notes for future edits

- `lib.core.filter_blocks` splits a sentence at its terminator, so
  `convert(sentence_file, sentence)` receives each fragment **without** the
  period. The old test `not re.search(r'\w$', part)` therefore fired on «,», «:»
  and «—» too. Never infer "sentence end" from the shape of a fragment.
- `_terminator_after(sentence, part)` looks the fragment up in the original
  sentence and checks whether what follows starts with `.`/`!`/`?`/`…` (or `..`).
  Only a real terminator pauses; comma/colon/dash never do.
- `_raw_source(part)` recovers the pre-accent form: it walks `_convert_parts`
  (the plain list of strings from `_split_sentence_on_sml`) backwards, skipping
  the `[voice:]`/`[/voice]` tags, and matches either the exact text or its
  `lstrip('+')` copy (the fragment reaching the loop is the **stressed** one).
  `_convert_parts` is a list of plain strings — not `(kind, value)` tuples.
- `_silence_follows` is unchanged: it already treats an explicit `[break]`/`[pause]`
  as "silence is coming" and a `[voice]`/`[emotion]` tag as transparent.
- **The app must be restarted** after touching `silero.py`: the module is loaded
  once and a live process keeps the old code.

### How to check

`env -u LD_LIBRARY_PATH python_env/bin/python -m unittest discover -s tests` →
`TerminatorPauseTests` covers the recovered period pause, the comma/colon/dash
non-pause, and the trailing fragment's pause.

## Fork change: no session-cleanup notice in the local GUI

The alert «Примечание: если в течение 60 дн. … сессия будет очищена. … должны быть
включены cookie.» is no longer shown.

### Files touched

| File | Change |
| --- | --- |
| `lib/gradio.py` | `_change_gr_restore_session` no longer appends `msg_session_cleanup_notice`/`msg_cookies_required`; `show_alert` runs only for the shared-GUI limit. |

### Design notes for future edits

- The shared-GUI branch still reports `msg_shared_access_limit`. The legend keys
  (`msg_session_cleanup_notice`, `msg_cookies_required`) are left in all 18
  `lib/lang/legends_*.py` files untouched — only the local path stopped using them.
- `delete_unused_tmp_dirs(...)` still runs in the local branch; the notice was
  removed, not the cleanup.

## Fork change: console progress logs (`[sml] …`)

Steps announce themselves in the console: `— начали` before, and
`— закончили[: <итог>] за <секунды> с` (or `— ошибка …`) after.

### Files touched

| File | Change |
| --- | --- |
| `sml_extractor/__init__.py` | `log(message)` prints `[sml] …` with `flush=True`; `traced(name, describe=None)` wraps a function with the start/finish/failure lines. |
| `sml_extractor/pipeline.py` | `analyze_book`, `load_book`, `describe_characters`, `cast_characters`, `character_rows`, `build_sml` are decorated with `traced(...)`. |
| `lib/sml_wizard.py` | `from sml_extractor import log as _log`; `analyze`, `build`, `build_5min`, `apply`, `cancel`, `save`, `restore`, `set_voice`, `remove_character`, `_set_field` and `reassign_voice_by_gender_age` log their outcome. |
| `lib/gradio.py` | The «Анализ книги» and «Собрать SML» handlers print the button press before calling the wizard. |

### Design notes for future edits

- `traced` is a decorator **factory**; its `describe` maps the return value to the
  outcome text and is wrapped in `try/except` so a bad `describe` cannot break the
  call. `functools.wraps` keeps the name and docstring. Put the decorator directly
  above `def`, above the docstring.
- Everything is plain `print` to stdout (the user chose console logs, not GUI
  status strings). `flush=True` keeps the order when stdout is a pipe.
- A failure is logged and the exception propagates unchanged, so the existing
  `except WizardError`/`except Exception` handlers in `gradio.py` still decide the
  GUI message.

### How to check

`env -u LD_LIBRARY_PATH python_env/bin/python -m unittest discover -s tests` → 227
tests OK. In a run, every `[sml]` step shows a matched `начали`/`закончили` pair.

## Fork change: the sentence size cap is 1024 (`SENTENCE_MAX_CHARS`)

A block is cut into the pieces the engine speaks one at a time by
`lib.core.get_sentences`. Its cap used to be `max_chars / 2` — **91 characters
for «rus»** — which is smaller than an ordinary sentence: a normal sentence was
then force-split, and `_force_split_segment` cut it at an internal comma, so the
engine ended the sentence intonationally in the middle («одно предложение — два
блока»). The cap is now **1024**, so any realistic sentence stays in one block
and is never cut.

### Files touched

| File | Change |
| --- | --- |
| `lib/conf_models.py` | New `SENTENCE_MAX_CHARS = 1024` next to `sml_escape_tag`. |
| `lib/__init__.py` | The constant is imported from `.conf_models` and re-exported in `__all__`, so `lib.core` (which does `from lib import *`) sees it. |
| `lib/core.py` | `get_sentences` uses `max_chars = SENTENCE_MAX_CHARS`; `_force_split_segment` prefers a hard terminator, then a space, and only falls back to a soft punctuation when neither exists (a comma makes the engine end the sentence, so it must never be the preferred cut); the orphan-merge ceiling became `max_chars` instead of `max_chars + max_chars//2`, so merging cannot push a piece above the cap. |
| `tests/test_sentence_split.py` | New: 6 tests over `get_sentences` (a long ordinary sentence stays whole, the cap is 1024, a pathological sentence is still split and stays within the cap, no piece ends on a comma). |

### Design notes for future edits

- The cap is a **single module constant** `SENTENCE_MAX_CHARS`; do not reintroduce
  a `language_mapping[...]['max_chars'] / N` derivation here. `max_chars` at
  `lib/core.py` line ~1645 (`/1.5`) belongs to the **raw flattening** stage in
  `get_blocks` and is unrelated — it never reaches `get_sentences`.
- `_force_split_segment`'s search order is deliberate: (1) last **hard**
  punctuation in the window, (2) last **space**, (3) last **soft** punctuation,
  (4) hard cut at the cap. A soft punctuation is the thing the engine hears as a
  sentence end, so it must be the last resort before the raw cut.
- `punctuation_split_hard_set` **includes `—` (em dash)**, which is also used as
  an internal parenthetical dash; that is pre-existing behaviour, left as is.
- The GUI sentence editor (`lib/gradio.py` ~1594) still validates a *hand-edited*
  sentence against `max_chars / 1.5` and the field itself is limited to 500 chars;
  that is a per-sentence editor limit and is intentionally independent of the
  1024 automatic-split cap.
- The change applies to every language (and to the CJK path inside
  `get_sentences`, which shares the same local `max_chars`).

### How to check

`env -u LD_LIBRARY_PATH python_env/bin/python -m unittest discover -s tests` → 233
tests OK (227 before + 6 new). Reproduce without a model:

```python
from types import SimpleNamespace
from lib import core
core.context = SimpleNamespace(get_session=lambda sid: {
    'language': 'rus', 'translate_enabled': False, 'translate': None,
    'tts_engine': 'silero', 'is_gui_process': False})
core.get_sentences('s', 'По воле благословенного Императора, решившего отойти '
                       'от ратных дел, бразды правления этой беспримерной '
                       'кампанией были вручены Хорусу, примарху Легиона '
                       'Лунных Волков.')
```
→ one block (before: two, cut at the comma after «дел»).
