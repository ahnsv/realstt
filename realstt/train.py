import bentoml
import whisper

model = whisper.load_model("base")

saved_model = bentoml.pytorch.save_model("whisper", model, labels={
    "version": "v0.1.0"
}, signatures={"__call__": {"batchable": True, "batch_dim": 0}}, )

print(saved_model)
j