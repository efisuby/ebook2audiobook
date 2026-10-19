import os
from lib.conf import voices_dir
from lib.conf_models import TTS_ENGINES, default_engine_settings

# Silero's russian v5 checkpoints live on models.silero.ai as torch.package
# archives (not on the Hugging Face hub): the engine downloads the url below
# into tts_dir/silero/ on first use. v5_6_ru is the newest of the family and
# the only one with automatic stress, homographs, question and emphasis
# intonation; it carries the 5 speakers the engine maps cast voices onto.
models = {
    # "internal" is the key default_fine_tuned points at, so the GUI/CLI
    # fine-tuned fallback resolves for this engine as well.
    # "lang" is matched against session["language"] when the GUI builds the
    # model list (_update_gr_fine_tuned_list), so it must be either "multi" or
    # an E2A language code ("rus"): the ISO1 "ru" would leave the dropdown
    # empty and gradio would reject the "internal" default.
    "internal": {
        "lang": "rus",
        "repo": "https://models.silero.ai/models/tts/ru/v5_6_ru.pt",
        "sub": "",
        "voice": default_engine_settings[TTS_ENGINES['SILERO']]['voice'],
        "files": [],
        "samplerate": default_engine_settings[TTS_ENGINES['SILERO']]['samplerate']
    },
    "v5_5_ru": {
        "lang": "rus",
        "repo": "https://models.silero.ai/models/tts/ru/v5_5_ru.pt",
        "sub": "",
        "voice": default_engine_settings[TTS_ENGINES['SILERO']]['voice'],
        "files": [],
        "samplerate": default_engine_settings[TTS_ENGINES['SILERO']]['samplerate']
    },
    "v4_ru": {
        "lang": "rus",
        "repo": "https://models.silero.ai/models/tts/ru/v4_ru.pt",
        "sub": "",
        "voice": default_engine_settings[TTS_ENGINES['SILERO']]['voice'],
        "files": [],
        "samplerate": default_engine_settings[TTS_ENGINES['SILERO']]['samplerate']
    }
}
