import torch
from torchmetrics.text import CharErrorRate, WordErrorRate

from whisper_transcribe import WhisperASR
from text_normalize import normalize_sentence

class Evaluation:

    def __init__(self, device=None, language="german"):
        self.cer = CharErrorRate()
        self.wer = WordErrorRate()
        
        
        if device is None:
            whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            whisper_device = device
        
        self.asr = WhisperASR(
            processor_name="openai/whisper-large-v3",
            model_name= "openai/whisper-large-v3",
            task="transcribe",
            language=language,
            device=whisper_device)

    def __call__(self, audio, audio_samplerate, graphemes):
        
        transcription = self.asr(audio, audio_samplerate)

        reference = ' '.join(normalize_sentence(graphemes))
        
        cer_score = self.cer(transcription, reference)
        wer_score = self.wer(transcription, reference)
        
        return {
            "reference": reference,
            "transcription": transcription[0],
            "cer": cer_score.item(),
            "wer": wer_score.item(),
        }
