
from flask import Flask, request, make_response
from flask_cors import CORS
import json
import os
import replicate
from pydantic import BaseModel
import requests
import io
import base64
from dotenv import load_dotenv
import asyncio
import openai


def generate_response(message, status_code):
    response = make_response(message, status_code)
    response.headers['Access-Control-Allow-Origin'] = '*'  # Allow requests from all origins
    response.headers['Access-Control-Allow-Headers'] = '*'  # Allow all headers
    response.headers['Access-Control-Allow-Methods'] = '*'  # Allow all HTTP methods
    return response

load_dotenv()

def generate_prompts(eng_sentence, target_language):
    system_message = f"You are an expert {target_language} tutor. Translate the sentence from english to the {target_language}."  

    prompt = f"""
translate the sentence to {target_language}
respond in the following format:
{"{"}
    "{target_language}_sentence": "[sentence]",        
{"}"}
make sure to only respond with only the defined format and nothing else
sentence:
{eng_sentence}"""

    return prompt, system_message

async def generate_sentence(prompt, system_message):
  # language = 'japanese'
  # system_message = f"You are an expert {language} tutor. Make sure to generate sentences with only the words your student knows. Your student also wants to learn the word for 'purple'. So always include this word."

  # prompt = generate_prompt(word_list, language)

  completion = await asyncio.to_thread(
    openai.ChatCompletion.create,
    model="gpt-3.5-turbo",
    temperature=0.9,
    messages=[
      {
        "role": "system",
        "content": system_message
      },
      {
        "role": "user",
        "content": prompt
      },
    ],
  )

  # print('got completion', completion)

  completion = completion.choices[0]['message']['content']

  return completion

from flask import Flask, request
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST', 'OPTIONS'])
def hello_world():
    print('working...', flush=True)
    if request.method == 'OPTIONS':  # Responding to preflight CORS request
        print('in options', flush=True)
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', '*')
        return response
    print('in general response', flush=True)
    try:
        language = request.form.get('language')
        image_file = request.files.get('image')
        with open('image.png', 'wb') as f:
            f.write(image_file.read())
        image_file.seek(0)  # Reset the file pointer to the beginning
        # Now you can use the language and image variables in your function
        # return 'Received data!'

        # Convert the image to base64 string
        image_data = image_file.read()
        base64_encoded_image = base64.b64encode(image_data).decode('utf-8')

        output = replicate.run(
            "yorickvp/llava-13b:6bc1c7bb0d2a34e413301fee8f7cc728d2d4e75bfab186aa995f63292bda92fc",
            input={
                "prompt":
                "Describe this image in one brief sentence",
                "image": open("image.png", "rb")
            })

        # import pdb; pdb.set_trace()
        transcript = "".join(list(output))
        print('transcript-english', transcript, flush=True)   

        prompt, system_message = generate_prompts(transcript, language)


        tries = 0
        MAX_RETRIES = 3
        word_ratio = None
        tokenized_sentence = None
        while tries < MAX_RETRIES:
            try:
                sentence_raw = asyncio.run(generate_sentence(prompt, system_message))
                print(sentence_raw)
                sentence = json.loads(sentence_raw)
                # import pdb
                # pdb.set_trace()
                assert (set(sentence.keys()) == set(
                    [f'{language}_sentence'])), "bad keys" + str(sentence.keys())
                assert (len(sentence[f'{language}_sentence'])
                        < 200), "bad sentence length" + str(
                            len(sentence[f'{language}_sentence']))
                translated_sentence = sentence[f'{language}_sentence']
                break
            except Exception as e:
                print('Exception', e)
                print('retrying')
                tries += 1

        print('translated_sentence', translated_sentence, flush=True)

        apikey = os.getenv("ELEVEN_API_KEY")

        # voice_id = "zDCgkBjkHDxMxYY6gXBj"

        CHUNK_SIZE = 1024

        url = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        url = url.format(voice_id=os.getenv('ELEVEN_VOICE_ID'))

        print(url)

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": apikey
        }

        data = {
            "text": translated_sentence,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        response = requests.post(url, json=data, headers=headers, stream=True)
        print('11 labs response', response.status_code, flush=True)

        audio_stream = io.BytesIO()

        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                audio_stream.write(chunk)

        audio_stream.seek(0)

        audio_data_bytes = audio_stream.read()
        audio_stream.close()
        audio_data_base64 = base64.b64encode(audio_data_bytes).decode('utf-8')
        result = {
            'audio_bytes': audio_data_base64,
            'transcript': transcript,
            'transcript_translated': translated_sentence
        }         
    except Exception as e:
        print(e)
        return generate_response({'error': str(e)}, 500)
    return generate_response(result, 200)

    

