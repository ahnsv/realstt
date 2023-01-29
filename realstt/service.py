import io
import logging
import contextlib
import time

import whisper
import bentoml
from bentoml.io import File, JSON

from realstt.runnables.audio_transcribe import AudioTranscriber

model = whisper.load_model("base") # TODO: replace this with custom trained model from MLflow
runner = bentoml.Runner(AudioTranscriber, name="whisper",
                        runnable_init_params={"model": model})

svc = bentoml.Service(name="whisper_service", runners=[runner])

logger = logging.getLogger(f"bentoml")


@contextlib.contextmanager
def log_execution_time(namespace: str):
    start_time = time.time()
    yield
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Executed {namespace=} in {execution_time:.2f} seconds.")


@svc.api(input=File(), output=JSON())
async def predict(audio: io.BytesIO):
    with log_execution_time("writing temp file"):
        path = f"/tmp/audio.mp4"
        with open(path, "wb") as f:
            f.write(audio.read())

    with log_execution_time("inference"):
        transcript = await runner.transcribe_audio.async_run(path)
    transcript_text = transcript["text"]
    return transcript_text
