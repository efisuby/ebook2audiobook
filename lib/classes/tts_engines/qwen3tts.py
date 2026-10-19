from lib.classes.tts_engines.common.headers import *
from lib.classes.tts_engines.common.preset_loader import load_engine_presets
from lib.lang import legends

#sys.stderr = StdoutFilter(sys.stdout)

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
                for part in sentence_parts:
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
                                part_tensor = self._tensor_type(audio_part).unsqueeze(0)
                                if part_tensor.numel() == 0:
                                    error = 'part_tensor not valid'
                                    return False, error
                                emotion_gain = self.params.get('emotion_gain')
                                if emotion_gain and float(emotion_gain) != 1.0:
                                    part_tensor = part_tensor * float(emotion_gain)
                                self.audio_segments.append(part_tensor)
                                if not re.search(r'\w$', part, flags=re.UNICODE) and part[-1] != '—':
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
