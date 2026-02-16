

import re
from typing import Dict, Optional, cast
from loguru import logger
import numpy as np
from pydantic import BaseModel, Field
from abc import ABC
import os
import torch
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from chat_engine.contexts.session_context import SessionContext
from funasr import AutoModel

from engine_utils.directory_info import DirectoryInfo
from engine_utils.general_slicer import SliceContext, slice_data
from handlers.audio.enhanced.audio_handler_noise_reduction import AudioPreprocessor, NoiseProfiler



class ASRConfig(HandlerBaseConfigModel, BaseModel):
    model_name: str = Field(default="iic/SenseVoiceSmall")


class ASRContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config = None
        self.local_session_id = 0
        self.output_audios = []
        self.audio_slice_context = SliceContext.create_numpy_slice_context(
            slice_size=16000,
            slice_axis=0,
        )
        self.cache = {}

        self.dump_audio = True
        self.audio_dump_file = None
        if self.dump_audio:
            dump_file_path = os.path.join(DirectoryInfo.get_project_dir(),
                                          "dump_talk_audio.pcm")
            self.audio_dump_file = open(dump_file_path, "wb")
        self.shared_states = None


class HandlerASR(HandlerBase, ABC):
    def __init__(self):
        super().__init__()

        self.model_name = 'iic/SenseVoiceSmall'

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            name="ASR_Funasr",
            config_model=ASRConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, 24000))
        inputs = {
            ChatDataType.HUMAN_AUDIO: HandlerDataInfo(
                type=ChatDataType.HUMAN_AUDIO,
            )
        }
        outputs = {
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT,
                definition=definition,
            )
        }
        return HandlerDetail(
            inputs=inputs, outputs=outputs,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        if isinstance(handler_config, ASRConfig):
            self.model_name = handler_config.model_name       
            model_path = os.path.join(DirectoryInfo.get_models_dir(), handler_config.model_name)
            if os.path.exists(model_path):
                self.model_name = model_path
        logger.info(f"load model {self.model_name}")
        self.model = AutoModel(model=self.model_name, disable_update=True)

    def create_context(self, session_context, handler_config=None):
        if not isinstance(handler_config, ASRConfig):
            handler_config = ASRConfig()
        context = ASRContext(session_context.session_info.session_id)
        context.shared_states = session_context.shared_states
        return context
    
    def start_context(self, session_context, handler_context):
        pass

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):

        output_definition = output_definitions.get(ChatDataType.HUMAN_TEXT).definition
        context = cast(ASRContext, context)
        if inputs.type == ChatDataType.HUMAN_AUDIO:
            audio = inputs.data.get_main_data()
        else:
            return
        speech_id = inputs.data.get_meta("speech_id")
        if (speech_id is None):
            speech_id = context.session_id

        if audio is not None:
            audio = audio.squeeze()

            logger.info('audio in')
            for audio_segment in slice_data(context.audio_slice_context, audio):
                if audio_segment is None or audio_segment.shape[0] == 0:
                    continue
                context.output_audios.append(audio_segment)

        speech_end = inputs.data.get_meta("human_speech_end", False)
        if not speech_end:
            return

        # prefill remainder audio in slice context
        remainder_audio = context.audio_slice_context.flush()
        if remainder_audio is not None:
            if remainder_audio.shape[0] < context.audio_slice_context.slice_size:
                remainder_audio = np.concatenate(
                    [remainder_audio,
                     np.zeros(shape=(context.audio_slice_context.slice_size - remainder_audio.shape[0]))])
                context.output_audios.append(remainder_audio)
        output_audio = np.concatenate(context.output_audios)
        if context.audio_dump_file is not None:
            logger.info('dump audio')
            context.audio_dump_file.write(output_audio.tobytes())

        res = self.model.generate(input=output_audio, batch_size_s=10)
        logger.info(res)
        context.output_audios.clear()
        output_text = re.sub(r"<\|.*?\|>", "", res[0]['text'])
        if len(output_text) == 0:
            # 如果 ASR 识别结果为空，则需要重新开启vad
            context.shared_states.enable_vad = True
            return
        output = DataBundle(output_definition)
        output.set_main_data(output_text)
        output.add_meta('human_text_end', False)
        output.add_meta('speech_id', speech_id)
        yield output

        end_output = DataBundle(output_definition)
        end_output.set_main_data('')
        end_output.add_meta("human_text_end", True)
        end_output.add_meta("speech_id", speech_id)
        yield end_output

    def destroy_context(self, context: HandlerContext):
        pass
class EnhancedSenseVoiceASR:
    def __init__(self, config):
        # Initialize base ASR
        self.base_asr = SenseVoiceASR(config)
        
        # Initialize audio preprocessor
        self.preprocessor = AudioPreprocessor(config.get('AudioPreprocessing', {}))
        self.noise_profiler = NoiseProfiler(config.get('NoiseProfiler', {}))
        
        # Calibration state
        self.is_calibrated = False
        
    async def calibrate_noise(self, audio_stream, duration_ms=3000):
        """
        Perform initial noise calibration
        Call this when session starts
        """
        print("Calibrating noise profile. Please remain silent...")
        
        # Collect noise samples
        samples_needed = int(duration_ms * 16 / 1000)  # Assuming 16kHz
        noise_audio = []
        
        while len(noise_audio) < samples_needed:
            chunk = await audio_stream.read_async()
            noise_audio.extend(chunk)
        
        # Create noise profile
        import numpy as np
        noise_profile = self.noise_profiler.calibrate(
            np.array(noise_audio[:samples_needed])
        )
        
        self.is_calibrated = True
        print("Calibration complete!")
        
        return noise_profile
    
    async def transcribe(self, audio, is_silence=False):
        """Transcribe audio with noise reduction"""
        import numpy as np
        audio_array = np.array(audio)
        
        # Update noise profile during silence
        if is_silence and self.is_calibrated:
            self.noise_profiler.update_noise_profile(audio_array, True)
        
        # Preprocess audio
        noise_profile = self.noise_profiler.get_noise_profile()
        processed_audio = self.preprocessor.process(audio_array, noise_profile)
        
        # Transcribe
        transcription = await self.base_asr.transcribe_async(processed_audio)
        
        return transcription
