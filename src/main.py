"""
asdf
"""
import os
import openai

# set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# path to your audio file
audio_file_path = "data/file.mp3"

# read the audio file
with open(audio_file_path, "rb") as audio_file:
    audio_content = audio_file.read()

# use the Whisper model to transcribe the audio
response = openai.Audio.transcribe(
    model="whisper-1",
    file=audio_content,
    response_format="text"
)

# get the transcription text
transcription_text = response['text']

# print the transcription
print("Transcription:\n", transcription_text)

# save the transcription to a .txt file
output_file_path = "transcription_output.txt"
with open(output_file_path, "w") as output_file:
    output_file.write(transcription_text)

print(f"Transcription saved to {output_file_path}")
