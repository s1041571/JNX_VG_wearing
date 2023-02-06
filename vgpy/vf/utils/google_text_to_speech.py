# pip install google-cloud-texttospeech
# 到這邊安裝 OpenSSL Light就可以了 https://slproweb.com/products/Win32OpenSSL.html

import os
from google.cloud import texttospeech

API_KEY_DIR = 'vgpy/vf/config/google_textToSpeech_API/My First Project-c0e87adcd181.json' # 沒有設定 角色 的那個 金鑰
# API_KEY_DIR = 'D:/google_textToSpeech_API/My First Project-e2bd54538e46.json' # 有角色 的那個 金鑰

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = API_KEY_DIR

# Instantiates a client
client = texttospeech.TextToSpeechClient()

# Build the voice request, select the language code ("en-US") and the ssml
# voice gender ("neutral")
voice = texttospeech.VoiceSelectionParams(
    language_code="cmn-TW", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
    name="cmn-TW-Standard-A"
)


# speed = 1.0

# # Select the type of audio file you want returned
# audio_config = texttospeech.AudioConfig(
#     audio_encoding=texttospeech.AudioEncoding.MP3,
#     speaking_rate=speed
# )

# # Set the text input to be synthesized
# synthesis_input = texttospeech.SynthesisInput(text="這是C5E測試，我是有角色的哦")

# # Perform the text-to-speech request on the text input with the selected
# # voice parameters and audio file type
# response = client.synthesize_speech(
#     input=synthesis_input, voice=voice, audio_config=audio_config
# )

# # The response's audio_content is binary.
# with open("output_role.mp3", "wb") as out:
#     # Write the response to the output file.
#     out.write(response.audio_content)
#     print('Audio content written to file "output.mp3"')


def text_to_mp3_file(text, speed, file_name):
    # file_name example: output_role.mp3
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speed
    )

    synthesis_input = texttospeech.SynthesisInput(text=text)

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # The response's audio_content is binary.
    with open(file_name, "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
        print('Audio content written to file...', file_name)


