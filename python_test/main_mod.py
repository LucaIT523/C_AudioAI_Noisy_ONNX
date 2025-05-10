import math
import argparse
import torch
import soundfile as sf
import numpy as np
from torch.nn import functional as F
import onnxruntime as ort


# torch.manual_seed(0)

FRAME_SIZE = 480
STATES_FULL_SIZE = 45304
ATTEN_LIM_DB  = 0.0
HOP_SZIE = 480
FFT_SIZE = 960
SAMPLE_RATE = 48000

INPUT_NAMES = [
    'input_frame', 
    'states',
    'atten_lim_db'
]
OUTPUT_NAMES = [
    'enhanced_audio_frame', 'out_states', 'lsnr'
]


def load_audio(filepath):
    audio, sr = sf.read(filepath, dtype='float32')
    tensor = torch.from_numpy(audio)  # Convert to PyTorch tensor
    return tensor, sr

def main(args):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.optimized_model_filepath = './model.onnx'
    sess_options.intra_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    ort_session = ort.InferenceSession('./model.onnx', sess_options, providers=['CPUExecutionProvider'])

    device = args.device
    input_frame = torch.rand(FRAME_SIZE)
    states = torch.zeros(STATES_FULL_SIZE, device=device)
    atten_lim_db = torch.tensor(ATTEN_LIM_DB, device=device)


    inference_path = args.input_path
    input_audio, sr = load_audio(inference_path)
    orig_len = input_audio.shape[0]

    hop_size_divisible_padding_size = (HOP_SZIE - orig_len % HOP_SZIE) % HOP_SZIE
    orig_len += hop_size_divisible_padding_size
    input_audio = F.pad(input_audio, (0, FFT_SIZE + hop_size_divisible_padding_size))
    
    chunked_audio = torch.split(input_audio, FRAME_SIZE)
    output_frames = []

    for input_frame in chunked_audio:
        output_tensors = []

        input_fead = {INPUT_NAMES[0]:input_frame.detach().cpu().numpy(), INPUT_NAMES[1]:states.detach().cpu().numpy(), INPUT_NAMES[2]:atten_lim_db.detach().cpu().numpy()}
        ort_result = ort_session.run(OUTPUT_NAMES, input_fead)
        for x in ort_result:
            output_tensors.append(torch.from_numpy(x))

        enhanced_audio_frame = output_tensors[0]
        states = output_tensors[1]
        lsnr = output_tensors[2]
        output_frames.append(enhanced_audio_frame)

    enhanced_audio = torch.cat(output_frames).unsqueeze(0) # [t] -> [1, t] typical mono format

    d = FFT_SIZE - HOP_SZIE
    enhanced_audio = enhanced_audio[:, d : orig_len + d]  
    audio_data = np.squeeze(enhanced_audio)
    audio_data = audio_data.to(torch.float32)
    sf.write('output_file.wav', audio_data, sr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Denoising one audio with DF3 model using torch only'       
    )
    parser.add_argument(
        '--input-path', type=str, default='./noisy.wav', help='Path to input audio file'
    )
    parser.add_argument(
        '--device', type=str, default='cpu', choices=['cuda', 'cpu'], help='Device to run on'
    )
    main(parser.parse_args())