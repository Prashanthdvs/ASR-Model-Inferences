import whisper
import numpy as np
from timeit import default_timer as timer
import tensorflow as tf
import multiprocessing
from transformers import  WhisperProcessor


wtokenizer = WhisperProcessor.from_pretrained(r"C:\Users\damojipurapuv.d\Downloads\whispersmall")
tflite_model_path=r"C:\Users\damojipurapuv.d\Downloads\file_float16\content\whisper_f16_small.tflite"

# Create an interpreter to run the TFLite model
interpreter = tf.lite.Interpreter(tflite_model_path, num_threads=multiprocessing.cpu_count())

# Allocate memory for the interpreter
interpreter.allocate_tensors()
input_tensor = interpreter.get_input_details()[0]['index']
output_tensor = interpreter.get_output_details()[0]['index']
inference_start = timer()

# Calculate the mel spectrogram of the audio file
print(f'Calculating mel spectrogram...')
filename= r"C:\Users\damojipurapuv.d\Downloads\recording.wav"

# audio = whisper.audio.load_audio(filename)
# audio = whisper.audio.pad_or_trim(audio)
# mel_from_file = whisper.audio.log_mel_spectrogram(audio)

mel_from_file = whisper.audio.log_mel_spectrogram(filename)

# # Pad or trim the input data to match the expected input size
input_data = whisper.audio.pad_or_trim(mel_from_file, whisper.audio.N_FRAMES)
print(input_data)
# Add a batch dimension to the input data
input_data = np.expand_dims(input_data, 0)

# Run the TFLite model using the interpreter
print("Invoking interpreter ...")
interpreter.set_tensor(input_tensor, input_data)
interpreter.invoke()

# Get the output data from the interpreter
output_data = interpreter.get_tensor(output_tensor)
print(output_data)

# convert tokens to text
print("Converting tokens ...")
for tokens in output_data:
    transcript=wtokenizer.batch_decode(np.expand_dims(tokens, axis=0), skip_special_tokens=True, fp16=False)[0]
    print(transcript)
print("\nInference took {:.2f}s ".format(timer() - inference_start))
