from lib.classes.tts_engines.common.headers import *
from lib.classes.tts_engines.common.preset_loader import load_engine_presets
from lib.lang import legends

import logging

#sys.stderr = StdoutFilter(sys.stdout)

class _OpenEndPadWarningFilter(logging.Filter):
    # transformers warns from generation/utils.py (_prepare_special_tokens) whenever a
    # generate() call leaves pad_token_id unset, then sets it to eos: "Setting
    # `pad_token_id` to `eos_token_id`:2150 for open-end generation." The id lives in a
    # nested Qwen3-TTS generation config, and passing pad_token_id through
    # generate_voice_clone does not reach it, so the line repeats for every sentence.
    # Nothing is wrong: pad = eos is what transformers picks anyway. Drop only that
    # record, any other warning from the same logger still prints.
    def filter(self, record:logging.LogRecord)->bool:
        return 'for open-end generation' not in record.getMessage()


def _silence_open_end_pad_warning()->None:
    logger = logging.getLogger('transformers.generation.utils')
    if not any(isinstance(f, _OpenEndPadWarningFilter) for f in logger.filters):
        logger.addFilter(_OpenEndPadWarningFilter())


class Qwen3TTS(TTSUtils, TTSRegistry, name='qwen3tts'):

    # Qwen3-TTS expects the language as its English name (or "Auto").
    language_names = {
        'zho': 'Chinese',
        'eng': 'English',
        'deu': 'German',
        'ita': 'Italian',
        'por': 'Portuguese',
        'spa': 'Spanish',
        'jpn': 'Japanese',
        'kor': 'Korean',
        'fra': 'French',
        'rus': 'Russian'
    }

    def __init__(self, session:DictProxy):
        try:
            self.session = session
            self.cache_dir = tts_dir
            self.speaker = None
            self.tts_key = self.session['model_cache']
            self.resampler_cache = {}
            self.audio_segments = []
            self.models = load_engine_presets(self.session['tts_engine'])
            self.params = {'voice_clone_prompts': {}}
            self.language = self.session.get('language')
            self.language_iso1 = self.session.get('language_iso1')
            if self.session.get('translate_enabled'):
                if self.session.get('translate'):
                    self.language = self.session['translate']
                if self.session.get('translate_iso1'):
                    self.language_iso1 = self.session['translate_iso1']
            fine_tuned = self.session.get('fine_tuned')
            if fine_tuned not in self.models:
                error = legends['error_invalid_fine_tuned'].format(model=fine_tuned, models=list(self.models.keys()))
                raise ValueError(error)
            model_cfg = self.models[fine_tuned]
            for required_key in ('repo', 'samplerate'):
                if required_key not in model_cfg:
                    error = legends['error_fine_tuned_missing_key'].format(model=fine_tuned, key=required_key)
                    raise ValueError(error)
            self.params['samplerate'] = model_cfg['samplerate']
            self.model_path = model_cfg['repo']
            self.qwen_language = self.language_names.get(self.language, 'Auto')
            enough_vram = self.session['free_vram_gb'] > 4.0
            seed = 0
            #random.seed(seed)
            self.amp_dtype = self._apply_gpu_policy(enough_vram=enough_vram, seed=seed)
            self.device = devices['CUDA']['proc'] if self.session['device'] in [devices['CUDA']['proc'], devices['ROCM']['proc'], devices['JETSON']['proc']] else self.session['device']
            # Qwen3-TTS exposes no temperature/top_p surface on this API, so the
            # [emotion:...] tag only reaches it through the audio gain.
            self.fine_tuned_params = {}
            self.engine = self.load_engine()
        except Exception as e:
            error = f'__init__() error: {e}'
            raise ValueError(error)

    def load_engine(self)->Any:
        try:
            msg = legends['msg_loading_tts_model'].format(model=self.tts_key)
            print(msg)
            self.cleanup_memory()
            engine = loaded_tts.get(self.tts_key)
            if not engine:
                try:
                    import torch
                    from qwen_tts import Qwen3TTSModel
                except ImportError as e:
                    error = f'load_engine(): the qwen-tts package is required for {self.session["tts_engine"]}: {e}'
                    raise RuntimeError(error) from e
                on_gpu = self.device != devices['CPU']['proc']
                try:
                    engine = Qwen3TTSModel.from_pretrained(
                        self.model_path,
                        device_map=self.device if on_gpu else 'cpu',
                        dtype=torch.bfloat16 if on_gpu else torch.float32,
                        cache_dir=self.cache_dir
                    )
                except Exception as e:
                    error = f'load_engine(): failed to load {self.model_path}: {e}'
                    raise RuntimeError(error) from e
                _silence_open_end_pad_warning()
                loaded_tts[self.tts_key] = engine
            if engine:
                msg = legends['msg_tts_loaded'].format(model=self.tts_key)
                print(msg)
                return engine
            error = 'load_engine(): engine is None'
            raise RuntimeError(error)
        except Exception as e:
            error = f'load_engine() error: {e}'
            raise RuntimeError(error) from e

    def _trailing_silence(self)->float:
        # seconds of silence sitting at the end of self.audio_segments
        import torch
        total = 0.0
        for segment in reversed(self.audio_segments):
            if torch.count_nonzero(segment) == 0:
                total += segment.shape[-1] / self.params['samplerate']
            else:
                break
        return total

    @classmethod
    def _silence_follows(cls, sentence_parts:list[str], index:int)->bool:
        # True when the next thing that produces audio after sentence_parts[index] is an
        # explicit [break]/[pause] tag: those insert silence of their own, so the
        # punctuation pause must not be stacked on top of them.
        for nxt in sentence_parts[index + 1:]:
            m = SML_TAG_PATTERN.fullmatch(nxt)
            if m is None:
                # plain text is audible unless it is punctuation-only, which the render
                # loop skips for exactly the same reason (no alphanumeric characters)
                if any(c.isalnum() for c in nxt):
                    return False
                continue
            tag = m.group('tag')
            if tag in ('break', 'pause'):
                return True
            if tag not in ('voice', 'emotion'):
                return False
        return False

    def _voice_clone_prompt(self, ref_audio:str)->Any:
        prompts = self.params['voice_clone_prompts']
        if ref_audio in prompts:
            return prompts[ref_audio]
        prompt = self.engine.create_voice_clone_prompt(ref_audio=ref_audio, x_vector_only_mode=True)
        prompts[ref_audio] = prompt
        return prompt

    def convert(self, sentence_file:str, sentence:str, **kwargs)->tuple:
        try:
            import torch
            import numpy as np
            from lib.classes.tts_engines.common.audio import trim_audio, is_audio_data_valid
            if self.engine:
                sentence_parts = self._split_sentence_on_sml(sentence)
                self.params['block_voice'] = kwargs.get('block_voice', self.session['voice'])
                if self.params.get('inline_voice'):
                    self.params['current_voice'] = self.params['inline_voice']
                else:
                    self.params['current_voice'], error = self._set_voice(self.params['block_voice'])
                    if self.params['current_voice'] is None and error is not None:
                        return False, error
                self.audio_segments = []
                # the last fragment that carries words (same alnum test as below): it must
                # leave a pause behind it, otherwise two sentences would be glued together
                # once the per-fragment padding is trimmed away
                last_audible = max(
                    (i for i, p in enumerate(sentence_parts) if any(c.isalnum() for c in p.strip())),
                    default=-1
                )
                previous_voice = None
                for index, part in enumerate(sentence_parts):
                    part = part.strip()
                    if not part:
                        continue
                    if SML_TAG_PATTERN.fullmatch(part):
                        success, error = self._convert_sml(part)
                        if not success:
                            return False, error
                        continue
                    if not any(c.isalnum() for c in part):
                        continue
                    else:
                        if part.endswith("'"):
                            part = part[:-1]
                        ref_audio = self.params['current_voice']
                        part_sr = self.params['samplerate']
                        if previous_voice is not None and ref_audio != previous_voice:
                            # a speaker change is a pause of its own: top up whatever silence
                            # is already there to a short gap, so the new voice does not start
                            # on the heels of the previous one
                            gap = int(np.random.uniform(0.3, 0.5) * 100) / 100 - self._trailing_silence()
                            if gap > 0:
                                self.audio_segments.append(torch.zeros(1, int(self.params['samplerate'] * gap)))
                        try:
                            with torch.inference_mode():
                                try:
                                    prompt = self._voice_clone_prompt(ref_audio)
                                    wavs, part_sr = self.engine.generate_voice_clone(
                                        text=part,
                                        language=self.qwen_language,
                                        voice_clone_prompt=prompt
                                    )
                                except Exception as prompt_error:
                                    # x-vector-only cloning is unsupported for some
                                    # reference wavs; fall back to the inline pair.
                                    print(f'{self.__class__.__name__}: clone prompt failed ({prompt_error}), retrying with ref_audio')
                                    wavs, part_sr = self.engine.generate_voice_clone(
                                        text=part,
                                        language=self.qwen_language,
                                        ref_audio=ref_audio
                                    )
                            audio_part = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
                            if audio_part is not None and is_audio_data_valid(audio_part):
                                if part_sr and part_sr != self.params['samplerate']:
                                    audio_part = self._resample_audiodata(audio_part, part_sr, self.params['samplerate'])
                                part_tensor = self._tensor_type(audio_part)
                                # Qwen3-TTS pads every fragment it generates (200-500 ms of
                                # leading silence and as much trailing); inside a multi-voice
                                # sentence those paddings piled up with the [break]/[pause]/
                                # punctuation silences into seconds of dead air.
                                trimmed = trim_audio(part_tensor, self.params['samplerate'])
                                if trimmed.numel() == 0:
                                    trimmed = part_tensor
                                part_tensor = trimmed.unsqueeze(0)
                                if part_tensor.numel() == 0:
                                    error = 'part_tensor not valid'
                                    return False, error
                                emotion_gain = self.params.get('emotion_gain')
                                if emotion_gain and float(emotion_gain) != 1.0:
                                    part_tensor = part_tensor * float(emotion_gain)
                                self.audio_segments.append(part_tensor)
                                previous_voice = ref_audio
                                # a fragment that does not end on a word gets a pause of its own,
                                # but only when nothing else already pauses for it: stacking the
                                # punctuation pause on top of a [break]/[pause] (or on the last
                                # fragment, whose trailing padding is trimmed away) is what made
                                # the switches sound like long gaps.
                                if not self._silence_follows(sentence_parts, index) and (
                                    index == last_audible
                                    or (not re.search(r'\w$', part, flags=re.UNICODE) and part[-1] != '—')
                                ):
                                    silence_time = int(np.random.uniform(0.3, 0.6) * 100) / 100
                                    self.audio_segments.append(torch.zeros(1, int(self.params['samplerate'] * silence_time)))
                            else:
                                error = 'audio_part not valid'
                                return False, error
                        except IndexError as e:
                            self.cleanup_memory()
                            error = f'convert() error at {e} segment: {part}'
                            return False, error
                        except Exception as e:
                            # free the failed part's tensors before returning False;
                            # core.py then unloads the engine via unload_tts_manager().
                            self.cleanup_memory()
                            return False, self.log_exception(f'{self.__class__.__name__}.convert() part loop', e)
                if self.audio_segments:
                    segment_tensor = torch.cat(self.audio_segments, dim=-1)
                    if not self.audio_save(sentence_file, segment_tensor, self.params['samplerate']):
                        error = f'audio_save() error: cannot save {sentence_file}'
                        return False, error
                    self.audio_segments = []
                    if not os.path.exists(sentence_file):
                        error = legends['error_cannot_create'].format(file=sentence_file)
                        return False, error
                return True, None
            else:
                error = legends['error_tts_engine_load_failed'].format(engine=self.session['tts_engine'])
                return False, error
        except Exception as e:
            self.cleanup_memory()
            self.audio_segments = []
            return False, self.log_exception(f'{self.__class__.__name__}.convert()',e)

    def create_vtt(self, all_sentences:list)->bool:
        if self._build_vtt_file(all_sentences):
            return True
        return False
