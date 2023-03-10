import bentoml
import whisper
import torch


class AudioTranscriber(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self, model: whisper.Whisper):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @bentoml.Runnable.method(batchable=False)
    def transcribe_audio(self, audio_path):
        return self.model.transcribe(audio_path)
