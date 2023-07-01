import whisper
import numpy as np
from timeit import default_timer as timer
import tensorflow as tf
from transformers import WhisperTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperProcessor
import librosa

# Define the path to the TFLite model
tflite_model_path = r"C:\Users\damojipurapuv.d\Downloads\Prashanth\tflite_model_sample\Whisper_small_model\whisper_f16_small.tflite"
#tflite_model_path = r"C:\Users\damojipurapuv.d\Downloads\Prashanth\tflite_model_sample\whisper-base1.tflite"
processor = WhisperProcessor.from_pretrained(r"C:\Users\damojipurapuv.d\Downloads\whispersmall")
interpreter = tf.lite.Interpreter(tflite_model_path)
inference_start = timer()
interpreter.allocate_tensors()

#Get the input and output tensors
input_tensor = interpreter.get_input_details()[0]['index']
output_tensor = interpreter.get_output_details()[0]['index']
inference_start = timer()

# Calculate the mel spectrogram of the audio file
print(f'Calculating mel spectrogram...')
audio = whisper.load_audio(r"C:\Users\damojipurapuv.d\Downloads\Saloon HVAC control_1_.wav")
audio = whisper.pad_or_trim(audio)
mel_from_file = whisper.audio.log_mel_spectrogram(audio)

# Pad or trim the input data to match the expected input size
input_data = whisper.audio.pad_or_trim(mel_from_file, whisper.audio.N_FRAMES)
input_data = np.array(input_data, dtype=np.int64)
# Add a batch dimension to the input data
input_data = np.expand_dims(input_data, 0)       #dtype=np.uint8

# Run the TFLite model using the interpreter
print("Invoking interpreter ...")
interpreter.set_tensor(input_tensor, input_data)
interpreter.invoke()

# Get the output data from the interpreter
output_data = interpreter.get_tensor(output_tensor)
print(output_data)
wtokenizer = WhisperProcessor.from_pretrained('Shubham09/LISA_Whisper_small_latest')

# convert tokens to text
print("Converting tokens ...")
for tokens in output_data:
    text=wtokenizer.batch_decode(np.expand_dims(tokens, axis=0), skip_special_tokens=True)[0]
    print(text)
print("\nInference took {:.2f}s ".format(timer() - inference_start)) 


# filename = r"C:\Users\damojipurapuv.d\Downloads\Saloon HVAC control_1_.wav"
# speech, _ = librosa.load(filename)
# inference_start = timer()
# print("going to interpreter")
# tflite_generate = interpreter.get_signature_runner()
# input_features = processor(speech, return_tensors="pt", dtype=tf.int8).input_features
# print(input_features)
# generated_ids = tflite_generate(input_features=input_features)["sequences"]
# transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(transcript)