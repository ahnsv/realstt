import io

from fastapi import FastAPI, WebSocket

app = FastAPI()


@app.websocket("/audio")
async def audio(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            message = await ws.receive()
            audio_blob = io.BytesIO(message['bytes'])
            print(audio_blob.getvalue())
    except Exception as e:
        raise Exception(f'Could not process audio: {e}')
    finally:
        await ws.close()
