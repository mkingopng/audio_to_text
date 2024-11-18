"""
asdf
"""
import whisper

# Load the model
model = whisper.load_model("base")

# Transcribe the audio file
result = model.transcribe("./../data/jordan-2024-09-12-04-11-20.m4a")

# Get the transcription text
transcription_text = result['text']

# Print the transcription
print("Transcription:\n", transcription_text)

# Save the transcription to a .txt file
output_file_path = "../data/jordan-2024-09-12-04-11-20.txt"
with open(output_file_path, "w") as output_file:
    output_file.write(transcription_text)

print(f"Transcription saved to {output_file_path}")
