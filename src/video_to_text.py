"""
transcribe the audio from a video file and save the transcription to a text file
"""
import whisper
import os
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.utils import make_chunks
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Load the Whisper model
model = whisper.load_model("base")

# Load the MP4 file using moviepy
video = VideoFileClip("../data/ZeroW_Staff_Training_2.mp4")

# Extract audio from the video and save it to a temporary file
temp_audio_path = "temp_audio.wav"
video.audio.write_audiofile(temp_audio_path)

# Load the audio file with pydub for chunking if necessary
audio_segment = AudioSegment.from_file(temp_audio_path, format="wav")

# Split audio into chunks
chunk_length_ms = 60000  # 1 minute chunks
chunks = make_chunks(audio_segment, chunk_length_ms)

# Transcribe each chunk and combine results
transcripts = []

for i, chunk in enumerate(tqdm(chunks, desc="Transcribing chunks")):
    chunk_name = f"chunk{i}.wav"
    chunk.export(chunk_name, format="wav")
    result = model.transcribe(chunk_name)
    transcripts.append(result["text"])
    # Clean up chunk file after transcription
    os.remove(chunk_name)

# Combine all transcripts into a single file
with open("transcribed_text.txt", "w") as f:
    for transcript in transcripts:
        f.write(transcript + "\n")

# Clean up the temporary audio file
os.remove(temp_audio_path)
