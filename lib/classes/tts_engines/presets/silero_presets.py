import os
from lib.conf import voices_dir
from lib.conf_models import TTS_ENGINES, default_engine_settings

# Silero's russian checkpoints live on models.silero.ai as torch.package archives
# (not on the Hugging Face hub): the engine downloads the url below into
# tts_dir/silero/ on first use. They carry an accentor and a homograph solver of
# their own and advertise the 5 speakers the engine maps the cast voices onto;
# v5_6_ru is the newest of the family (stress, homographs, question and emphasis
# intonation).
#
# Per entry:
#   lang    matched against session["language"] by the GUI model list
#           (_update_gr_fine_tuned_list), so it must be either "multi" or an E2A
#           language code ("rus"): the ISO1 "ru" would leave the dropdown empty
#           and gradio would then reject the "internal" default.
#   repo    download url of the torch.package checkpoint.
#   stress  run silero-stress over every fragment before the synthesis
#           (optional, default True): the checkpoints honour the '+' marks it
#           writes, and its accentor/homograph solver is newer than the one
#           packed into them. Set it to False to synthesize the raw text.
models = {
    # "internal" is the key default_fine_tuned points at, so the GUI/CLI
    # fine-tuned fallback resolves for this engine as well.
    "internal": {
        "lang": "rus",
        "repo": "https://models.silero.ai/models/tts/ru/v5_6_ru.pt",
        "sub": "",
        "voice": default_engine_settings[TTS_ENGINES['SILERO']]['voice'],
        "files": [],
        "samplerate": default_engine_settings[TTS_ENGINES['SILERO']]['samplerate'],
        "stress": True
    },
    "v5_5_ru": {
        "lang": "rus",
        "repo": "https://models.silero.ai/models/tts/ru/v5_5_ru.pt",
        "sub": "",
        "voice": default_engine_settings[TTS_ENGINES['SILERO']]['voice'],
        "files": [],
        "samplerate": default_engine_settings[TTS_ENGINES['SILERO']]['samplerate'],
        "stress": True
    },
    "v4_ru": {
        "lang": "rus",
        "repo": "https://models.silero.ai/models/tts/ru/v4_ru.pt",
        "sub": "",
        "voice": default_engine_settings[TTS_ENGINES['SILERO']]['voice'],
        "files": [],
        "samplerate": default_engine_settings[TTS_ENGINES['SILERO']]['samplerate'],
        "stress": True
    }
}
