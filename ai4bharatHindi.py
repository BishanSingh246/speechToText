""" Import Packages -"""
from transformers import AutoModelForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM, pipeline
import torchaudio
import torch
from datasets import load_dataset

# from IPython.display import Audio, display
import sys
import re
import pandas as pd
import time

# for creating csv file 
data = {
    "Index": [],
    "Audio": [],
    "Total Time": [],
    "Output Text": [],
    "Device": []
}


""" Function for loading Audio from File"""
def load_audio_from_file(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    num_channels, _ = waveform.shape
    if num_channels == 1:
        return waveform[0], sample_rate
    else:
        raise ValueError("Waveform with more than 1 channels are not supported.")


"""Load from file"""
# audio_path = 'output_audio0.wav'
# Specify the Hugging Face Model Id
model_id = "ai4bharat/indicwav2vec-hindi"
# Specify the Device Id on where to put the model
device_id = "cuda" if torch.cuda.is_available() else "cpu"
print("Device :", device_id)
# Specify Decoder Type:
decoderType = "greedy" # Choose "LM" decoding or "greedy" decoding
# Load Model
model_instance = AutoModelForCTC.from_pretrained(model_id).to(device_id)

if decoderType == "greedy":
    # Load Processor without language model
    processor = Wav2Vec2Processor.from_pretrained(model_id)
else:
    # Load Processor with language model
    processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_id)

def runSTT(index,audio_path):
    At_index = f"At index : {index} "
    print(At_index)
    start = time.time()

    targetSampleRate = 16000

    #Load from file
    waveform, sample_rate = load_audio_from_file(audio_path)

    #ResampleTARGET_SAMPLE_RATE = 16000
    resampled_audio = torchaudio.functional.resample(waveform, sample_rate, targetSampleRate)
    
    """ Process Audio Data and Run Forward Pass to obtain Logits"""
    # Process audio data
    input_tensor = processor(resampled_audio, return_tensors="pt", sampling_rate=targetSampleRate).input_values

    # Run forward pass
    with torch.no_grad():
        logits = model_instance(input_tensor.to(device_id)).logits.cpu()

    """# Decode Logits"""

    if decoderType == "greedy":
        prediction_ids = torch.argmax(logits, dim=-1)
        output_str = processor.batch_decode(prediction_ids)[0]
        print(f"Greedy Decoding: {output_str}")
    else:
        output_str = processor.batch_decode(logits.numpy()).text[0]
        print(f"LM Decoding: {output_str}")
    end = time.time()
    total_time = end-start
    # total time taken
    print("---------------------------------------------------------------------------------")
    print("audio file - ",audio_path)
    print("Execution time of the program is- ", total_time)
    print("------------------------------------------------------------------------------------------")
    data["Index"].append(index)
    data["Audio"].append(audio_path)
    data["Total Time"].append(total_time)
    data["Output Text"].append(output_str)
    data["Device"].append(device_id)

fileName = "hindi_16gb_cpu_ai4bharat"

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
    runSTT(index, value)
print(f"Data writing to {fileName}.csv file...")
# Create a DataFrame from the data dictionary
df = pd.DataFrame(data)

# Write the DataFrame to a CSV file
df.to_csv(f"{fileName}.csv", index=False, encoding="utf-8")

print(f"Data written to {fileName}.csv file...")
