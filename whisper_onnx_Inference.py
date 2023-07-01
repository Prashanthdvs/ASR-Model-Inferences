#ASR offline whisper model inference in ONNX model format


from transformers import AutoProcessor, pipeline
from optimum.onnxruntime.modeling_seq2seq import ORTModelForSpeechSeq2Seq  #, ORTModelForConditionalGeneration
import librosa
from optimum.onnxruntime import ORTModel  #, ORTModelForConditionalGeneration
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperForConditionalGeneration
from transformers import TFWhisperModel


file = r"C:\Users\damojipurapuv.d\Downloads\audio test\audio test\Audio3.wav"
save_directory = r"C:\Users\damojipurapuv.d\Downloads\Prashanth\tflite_model_sample\new_onnx1"
model = ORTModelForSpeechSeq2Seq.from_pretrained("Quantified_onnx_small")
processor = WhisperProcessor.from_pretrained("Quantified_onnx_small")
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = "en", task = "transcribe")
speech_array, sr = librosa.load(file)
input_features = processor(speech_array, return_tensors="pt", sr=16000).input_features 
predicted_ids = model.generate(input_features)
transcript = processor.batch_decode(predicted_ids,skip_special_tokens = True)[0]
print(transcript)










