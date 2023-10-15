import json
import os
import replicate
from pydantic import BaseModel
import requests
import io
import base64
from dotenv import load_dotenv

_11_labs_key = "74e45ff2747a2f8be872f3218adf4ac3"

load_dotenv()
output = replicate.run(
    "yorickvp/llava-13b:6bc1c7bb0d2a34e413301fee8f7cc728d2d4e75bfab186aa995f63292bda92fc",
    input={
        "prompt":
        "In french language describe this image in one brief sentence",
        "image": open("dog.png", "rb")
    })

transcript = "".join(list(output))
print(transcript)

apikey = os.getenv("ELEVEN_API_KEY")

voice_id = "zDCgkBjkHDxMxYY6gXBj"

CHUNK_SIZE = 1024

url = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
url = url.format(voice_id=voice_id)

print(url)

headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": apikey
}

data = {
    "text": transcript,
    "model_id": "eleven_multilingual_v2",
    "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.5
    }
}

response = requests.post(url, json=data, headers=headers, stream=True)
print(response.status_code)

audio_stream = io.BytesIO()

for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
  if chunk:
    audio_stream.write(chunk)

audio_stream.seek(0)

audio_data_bytes = audio_stream.read()
audio_stream.close()
audio_data_base64 = base64.b64encode(audio_data_bytes).decode('utf-8')
audio_data_json = {'audio_bytes': audio_data_base64}