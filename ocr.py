import pytesseract
from PIL import Image
import re
import soundfile as sf
from TTS.api import TTS

# Initialize Coqui TTS model
tts_model = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False)

def extract_text(image_path):
    """Extracts and cleans text from an image using Tesseract OCR."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)

    # Clean OCR text
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces and newlines
    return text

def save_speech(text, filename="output.wav"):
    """Converts text to speech using Glow-TTS and saves it as a WAV file."""
    if not text:
        print("No text detected!")
        return

    # Ensure minimum text length for TTS
    words = text.split()
    if len(words) < 5:
        print("Input text is too short; adding filler words.")
        text += " This is a sample text for processing."

    print(f"\n🔹 Saving speech to: {filename}")

    # Generate speech
    audio_data = tts_model.tts(text)

    # Save to file
    sf.write(filename, audio_data, samplerate=22050)
    print(f"✅ Audio saved successfully: {filename}")

# 📌 Provide image path here
image_path = "/home/sag_umt/Downloads/book.jpeg"  # Replace with your image file

# Extract text and convert to speech
extracted_text = extract_text(image_path)
print("\nExtracted Text:\n", extracted_text)

# Save extracted text as speech
save_speech(extracted_text)
