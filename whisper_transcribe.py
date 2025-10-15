from transformers import WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2CTCTokenizer
import torchaudio.functional as F
import torch
from numpy.typing import ArrayLike
from typing import List

from text_normalize import NormalizeSentence

class AudioNormalizer(torch.nn.Module):
    def __init__(
        self,
        output_samplerate,
        squeeze_channels=False,
    ):
        super().__init__()
        self.output_samplerate = output_samplerate
        self.squeeze_channels = squeeze_channels

    def _to_float(
        self,
        waveform: ArrayLike,
    ) -> torch.Tensor:
        """
        Convert any audio array to Torch float tensor.
        :param waveform: Audio to convert.
        :return: Audio as float.
        """
        # Convert to torch tensor
        if not isinstance(
            waveform,
            torch.Tensor,
        ):
            waveform = torch.tensor(waveform)
        # Convert to float
        if not waveform.dtype == torch.float32:
            input_dtype = waveform.dtype
            waveform = waveform.float()
            if input_dtype == torch.int16:
                waveform = waveform / 32768.0
        return waveform

    def _resample_like_librosa(
        self,
        waveform: ArrayLike,
        input_samplerate: int,
        output_samplerate: int,
    ) -> torch.Tensor:
        """
        Resample audio with results similar librosa
        with 'kaiser best' setting.
        :param waveform: Audio to resample.
        :param input_samplerate: Input sample rate.
        :param output_samplerate: Output sample rate.
        :return: Resampled audio.
        """
        waveform = self._to_float(waveform)
        if input_samplerate != output_samplerate:
            waveform = F.resample(
                waveform=waveform,
                orig_freq=input_samplerate,
                new_freq=output_samplerate,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492,
            )
        return waveform

    def forward(
        self,
        x,
        input_samplerate,
    ):
        if self.squeeze_channels:
            x = x.squeeze(0)

        x = self._resample_like_librosa(
            waveform=x,
            input_samplerate=input_samplerate,
            output_samplerate=self.output_samplerate,
        )
        return (
            x,
            self.output_samplerate,
        )
        
class WhisperASR(torch.nn.Module):
    def __init__(
        self,
        processor_name,
        model_name,
        task: str,
        language: str,
        normalize_transcriptions: bool = True,        
        device: str = "cpu",
    ):
        super().__init__()

        self.model_device = device
        
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.processor = WhisperProcessor.from_pretrained(processor_name)
        self.model.config.forced_decoder_ids = None
        self.model.to(self.model_device)
        
        self.decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language, task=task
        )
        self.normalize_transcriptions = normalize_transcriptions
        self.normalize_sentence = NormalizeSentence()
        self.audio_normalizer = None
        self.audio_normalizer = AudioNormalizer(
            output_samplerate=16000, squeeze_channels=True
        )

    def forward(self, x: torch.Tensor, input_fs: int) -> List[str]:
        """
        Args:
            x (torch.Tensor): batch of audio samples [B, 1, T] or [B, T]
        """
        assert (
            x.dim() == 2 or x.dim() == 3
        ), f"Expected 2 or 3 dims for x, got {x.dim()}"
        if x.dim() == 3:
            assert x.size(1) == 1, f"Expected 1 channel in x, got {x.size(1)}"
            x = x.squeeze(1)

        audio = [
            self.audio_normalizer(xx, input_fs)[0].cpu().to(torch.float32).numpy()
            for xx in x
        ]
        input_fts = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features

        input_fts = input_fts.to(self.model_device)
        out = self.model.generate(input_fts, forced_decoder_ids=self.decoder_ids, temperature=0.0)
        out = out.to('cpu')
        transcriptions = self.processor.batch_decode(out, skip_special_tokens=True)

        if self.normalize_transcriptions:
            transcriptions = self.normalize_sentence(transcriptions)

        return transcriptions


if __name__ == "__main__":
    import argparse
    import torchaudio
    
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=str,  help="Path to audio file to transcribe")
    parser.add_argument("--model_name", type=str, default="openai/whisper-large-v3", help="Whisper model name")
    parser.add_argument("--language", type=str, default="german", help="Language spoken in the audio")
    args = parser.parse_args()
    
    asr = WhisperASR(
        processor_name=args.model_name,
        model_name=args.model_name,
        language=args.language,
        task="transcribe",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    speech, sr = torchaudio.load(args.audio)
    print(asr(speech, sr))
        
    