import tkinter as tk
from tkinter import Toplevel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import onnx
import torch
import onnxruntime as ort
import sounddevice as sd
from scipy.io.wavfile import write, read
from mltu.preprocessors import WavReader
from mltu.utils.text_utils import get_cer, get_wer
import numpy as np

from utils import read_batch, split_into_batches, prepare_model_input, getLabels
from utils import Decoder

# Load the ONNX model
onnx_model = 'Models/05_sound_to_text/202411180003/model.onnx'

# Initialize the Tkinter window
window = tk.Tk()
window.title("Speech to Text with ML Model")
window.geometry("800x600")  # Main window size

# Global variables
recording = False
audio_buffer = []  # Initialize the audio buffer
transcriptions = []  # List to store transcription history
labelTrue = "i really like dogs i have one at home"
temp_audio = "temp_audio.wav"
sample_rate = 16000

# Define functions for recording and transcribing
def toggle_record_audio():
    global recording, audio_buffer
    if not recording:
        # Start recording
        recording = True
        status_label.config(text="Recording...")
        record_button.config(text="Stop Recording")
        audio_buffer = []  # Reset the audio buffer
        stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', callback=audio_callback)
        stream.start()
    else:
        # Stop recording
        recording = False
        status_label.config(text="Recording complete")
        record_button.config(text="Record Audio")
        
        # Save audio file
        write(temp_audio, sample_rate, np.array(audio_buffer, dtype='int16'))
        process_audio(temp_audio)

# Callback to capture audio into a buffer
def audio_callback(indata, frames, time, status):
    global audio_buffer
    if status:
        print(f"Status: {status}")
    audio_buffer.extend(indata.copy().flatten())

def process_audio(temp_audio):
    global transcriptions
    test_file = [temp_audio]
    batches = split_into_batches(test_file, batch_size=10)
    input = prepare_model_input(read_batch(batches[0]))

    # ONNX inference and decoding
    labels = getLabels()
    decoder_instance = Decoder(labels)

    ort_session = ort.InferenceSession(onnx_model)
    onnx_input = input.detach().cpu().numpy()
    ort_inputs = {'input': onnx_input}
    ort_outs = ort_session.run(None, ort_inputs)
    transcription = decoder_instance(torch.Tensor(ort_outs[0])[0])

    transcriptions.append(transcription)  # Add to transcription history
    transcription_history.insert(tk.END, transcription)  # Update the Listbox
    update_transcription(transcription)
    recalculate_metrics(transcription)

def update_transcription(transcription):
    transcription_text.config(state=tk.NORMAL)
    transcription_text.delete(1.0, tk.END)
    transcription_text.insert(tk.END, transcription)
    transcription_text.config(state=tk.DISABLED)

def recalculate_metrics(transcription):
    global labelTrue
    labelTrue = labelTrue_entry.get()
    cer = get_cer(transcription, labelTrue)
    wer = get_wer(transcription, labelTrue)

    cer_label.config(text=f"Character Error Rate (CER): {cer:.3f}")
    wer_label.config(text=f"Word Error Rate (WER): {wer:.3f}")


def show_transcription(event):
    # Display the selected transcription from history
    selected_index = transcription_history.curselection()
    if selected_index:
        selected_transcription = transcriptions[selected_index[0]]
        update_transcription(selected_transcription)

# Create the GUI elements
status_label = tk.Label(window, text="Click 'Record' to start", font=('Arial', 12))
status_label.pack(pady=10)

record_button = tk.Button(window, text="Record Audio", command=toggle_record_audio, font=('Arial', 14))
record_button.pack(pady=10)

transcription_label = tk.Label(window, text="Transcription:", font=('Arial', 12))
transcription_label.pack(pady=5)

transcription_text = tk.Text(window, height=5, width=50, wrap=tk.WORD, font=('Arial', 12))
transcription_text.config(state=tk.DISABLED)
transcription_text.pack(pady=10)

history_label = tk.Label(window, text="Transcription History:", font=('Arial', 12))
history_label.pack(pady=5)

transcription_history = tk.Listbox(window, height=10, width=60, font=('Arial', 12))
transcription_history.bind('<<ListboxSelect>>', show_transcription)  # Bind selection to function
transcription_history.pack(pady=10)

labelTrue_label = tk.Label(window, text="Enter True Label:", font=('Arial', 12))
labelTrue_label.pack(pady=5)

labelTrue_entry = tk.Entry(window, font=('Arial', 12), width=40)
labelTrue_entry.insert(0, labelTrue)  # Default labelTrue value
labelTrue_entry.pack(pady=5)

update_metrics_button = tk.Button(window, text="Recalculate Metrics", font=('Arial', 14))
update_metrics_button.pack(pady=10)

cer_label = tk.Label(window, text="Character Error Rate (CER): N/A", font=('Arial', 12))
cer_label.pack(pady=5)

wer_label = tk.Label(window, text="Word Error Rate (WER): N/A", font=('Arial', 12))
wer_label.pack(pady=5)

# Start the Tkinter event loop
window.mainloop()
