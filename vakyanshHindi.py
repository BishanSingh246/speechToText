"""# Vakyansh hugging face code"""

import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor,Wav2Vec2ProcessorWithLM
import sox
import subprocess
import time
import re
import pandas as pd
# import gradio as gr


device_id = "cuda" if torch.cuda.is_available() else "cpu"
print("Device :", device_id)
# for creating csv file 
data = {
    "Index": [],
    "Audio": [],
    "Total Time": [],
    "Output Text": [],
    "Device": []
}


# Specify the Hugging Face Model Id
model_id = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"
processor = Wav2Vec2Processor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)

def read_file_and_process(wav_file):
    filename = wav_file.split('.')[0]
    filename_16k = filename + "16k.wav"
    resampler(wav_file, filename_16k)
    speech, _ = sf.read(filename_16k)
    print(speech)
    inputs = processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True)
    print(inputs)

    return inputs


def resampler(input_file_path, output_file_path):
    command = (
        f"ffmpeg -hide_banner -loglevel panic -i {input_file_path} -ar 16000 -ac 1 -bits_per_raw_sample 16 -vn "
        f"{output_file_path}"
    )
    subprocess.call(command, shell=True)

def parse_transcription(logits):
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    return transcription

def parse(index,audio_path):

    # record start time
    At_index = f"At index : {index} "
    print(At_index)
    start = time.time()
    input_values = read_file_and_process(audio_path)
    with torch.no_grad():
        logits = model(**input_values).logits
        # logits = model_instance(input_tensor.to(device_id)).logits.cpu()

    output_str = parse_transcription(logits)
    # record end time
    end = time.time()
    total_time = end-start
    
    print("---------------------------------------------------------------------------------")
    print("audio file - ",audio_path)
    print("Execution time of the program is- ", total_time)
    print("------------------------------------------------------------------------------------------")
    data["Index"].append(index)
    data["Audio"].append(audio_path)
    data["Total Time"].append(total_time)
    data["Output Text"].append(output_str)
    data["Device"].append(device_id)
    return output_str


fileName = "hindi_16gb_cpu_vakyansh"

audioPath=[
    "hindi/hindi1.wav",
    "hindi/hindi1.wav",
    "hindi/hindi1.wav",
    "hindi/hindi1.wav",
    "hindi/hindi1.wav",
    "hindi/hindi2.wav",
    "hindi/hindi2.wav",
    "hindi/hindi2.wav",
    "hindi/hindi2.wav",
    "hindi/hindi2.wav",
    "hindi/hindi3.wav",
    "hindi/hindi3.wav",
    "hindi/hindi3.wav",
    "hindi/hindi3.wav",
    "hindi/hindi3.wav",
    "hindi/hindi4.wav",
    "hindi/hindi4.wav",
    "hindi/hindi4.wav",
    "hindi/hindi4.wav",
    "hindi/hindi4.wav",
    "hindi/hindi5.wav",
    "hindi/hindi5.wav",
    "hindi/hindi5.wav",
    "hindi/hindi5.wav",
    "hindi/hindi5.wav",
    "hindi/hindi6.wav",
    "hindi/hindi6.wav",
    "hindi/hindi6.wav",
    "hindi/hindi6.wav",
    "hindi/hindi6.wav",
    "hindi/hindi7.wav",
    "hindi/hindi7.wav",
    "hindi/hindi7.wav",
    "hindi/hindi7.wav",
    "hindi/hindi7.wav",
    "hindi/hindi8.wav",
    "hindi/hindi8.wav",
    "hindi/hindi8.wav",
    "hindi/hindi8.wav",
    "hindi/hindi8.wav",
    "hindi/hindi9.wav",
    "hindi/hindi9.wav",
    "hindi/hindi9.wav",
    "hindi/hindi9.wav",
    "hindi/hindi9.wav",
    "hindi/hindi10.wav",
    "hindi/hindi10.wav",
    "hindi/hindi10.wav",
    "hindi/hindi10.wav",
    "hindi/hindi10.wav",
    "hindi/hindi11.wav",
    "hindi/hindi11.wav",
    "hindi/hindi11.wav",
    "hindi/hindi11.wav",
    "hindi/hindi11.wav",
    "hindi/hindi12.wav",
    "hindi/hindi12.wav",
    "hindi/hindi12.wav",
    "hindi/hindi12.wav",
    "hindi/hindi12.wav",
    "hindi/hindi13.wav",
    "hindi/hindi13.wav",
    "hindi/hindi13.wav",
    "hindi/hindi13.wav",
    "hindi/hindi13.wav",
    "hindi/hindi14.wav",
    "hindi/hindi14.wav",
    "hindi/hindi14.wav",
    "hindi/hindi14.wav",
    "hindi/hindi14.wav",
    "hindi/hindi15.wav",
    "hindi/hindi15.wav",
    "hindi/hindi15.wav",
    "hindi/hindi15.wav",
    "hindi/hindi15.wav"
    ]

for index, value in enumerate(audioPath):
    parse(index, value)
print(f"Data writing to {fileName}.csv file...")
# Create a DataFrame from the data dictionary
df = pd.DataFrame(data)

# Write the DataFrame to a CSV file
df.to_csv(f"{fileName}.csv", index=False, encoding="utf-8")

print(f"Data written to {fileName}.csv file...")

