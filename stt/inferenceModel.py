import tkinter as tk
from tkinter import messagebox
import onnx
import torch
import onnxruntime as ort
import sounddevice as sd
from scipy.io.wavfile import write
from mltu.preprocessors import WavReader
from mltu.utils.text_utils import get_cer, get_wer

from utils import read_batch, split_into_batches, read_audio, prepare_model_input, getLabels
from utils import Decoder

# Load the ONNX model
onnx_model = 'Models/05_sound_to_text/202411180003/model.onnx'

# Initialize the Tkinter window
window = tk.Tk()
window.title("Speech to Text with ML Model")

# Define functions for recording and transcribing
def record_audio():
    sample_rate = 16000
    duration = 5
    status_label.config(text="Recording...")
    # record the audio
    audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    status_label.config(text="Recording complete")
    # save the audio file
    temp_audio = "temp_audio.wav"
    write(temp_audio, sample_rate, audio)

    # process the audio and get transcription
    process_audio(temp_audio)

def process_audio(temp_audio):
    test_file = [temp_audio]
    batches = split_into_batches(test_file, batch_size=10)
    input = prepare_model_input(read_batch(batches[0]))


    

    # onnx inference and decoding
    #generate decoder_instance with the labels from the onnx file
    labels = getLabels()
    decoder_instance = Decoder(labels)

    ort_session = ort.InferenceSession(onnx_model)
    onnx_input = input.detach().cpu().numpy()
    ort_inputs = {'input': onnx_input}
    ort_outs = ort_session.run(None, ort_inputs)
    transcription = decoder_instance(torch.Tensor(ort_outs[0])[0])

    # get CER and WER
    label = "i really like dogs i have one at home"
    cer = get_cer(transcription, label)
    wer = get_wer(transcription, label)

    # update GUI with results
    transcription_text.config(state=tk.NORMAL)
    transcription_text.delete(1.0, tk.END)
    transcription_text.insert(tk.END, transcription)
    transcription_text.config(state=tk.DISABLED)

    cer_label.config(text=f"Character Error Rate (CER): {cer:.3f}")
    wer_label.config(text=f"Word Error Rate (WER): {wer:.3f}")

    # plot spectrogram and wave
    spectrogram = WavReader.get_spectrogram(temp_audio, frame_length=384, frame_step=256, fft_length=768)
    WavReader.plot_raw_audio(temp_audio, label)
    WavReader.plot_spectrogram(spectrogram, label)

# Create the GUI elements
status_label = tk.Label(window, text="Click 'Record' to start", font=('Arial', 12))
status_label.pack(pady=10)

record_button = tk.Button(window, text="Record Audio", command=record_audio, font=('Arial', 14))
record_button.pack(pady=10)

transcription_label = tk.Label(window, text="Transcription:", font=('Arial', 12))
transcription_label.pack(pady=5)

transcription_text = tk.Text(window, height=5, width=40, wrap=tk.WORD, font=('Arial', 12))
transcription_text.config(state=tk.DISABLED)
transcription_text.pack(pady=10)

cer_label = tk.Label(window, text="Character Error Rate (CER): N/A", font=('Arial', 12))
cer_label.pack(pady=5)

wer_label = tk.Label(window, text="Word Error Rate (WER): N/A", font=('Arial', 12))
wer_label.pack(pady=5)

# Start the Tkinter event loop
window.mainloop()
