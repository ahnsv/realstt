import io
import logging
import contextlib
import time

import whisper
import bentoml
from bentoml.io import File, JSON

from realstt.runnables.audio_transcribe import AudioTranscriber

model = whisper.load_model("base")
runner = bentoml.Runner(AudioTranscriber, name="whisper",
                        runnable_init_params={"model": model})

svc = bentoml.Service(name="whisper_service", runners=[runner])

logger = logging.getLogger(f"bentoml")


@contextlib.contextmanager
def log_execution_time():
    start_time = time.time()
    yield
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Executed in {execution_time:.2f} seconds.")


@svc.api(input=File(), output=JSON())
async def predict(audio: io.BytesIO):
    path = f"/tmp/audio.mp4"
    with open(path, "wb") as f:
        f.write(audio.read())

    with log_execution_time():
        transcript = await runner.transcribe_audio.async_run(path)
    transcript_text = transcript["text"]
    return transcript_text
