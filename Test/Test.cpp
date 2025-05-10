// Test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "ad_onnx.h"


int main()
{
    CAudioONNX_M                w_CAudioONNX_M;
    int                         w_nSts = AD_ERR_UNKNOWN;
    int                         w_nTotalFrameCount = 0;
    int                         w_nSampleRate = 0;
    int                         w_nChanel = 0;
    torch::Tensor               w_tsInAudio;
    torch::Tensor               w_tsInAudio_FFT;
    std::vector<torch::Tensor>  output_frames;
    torch::Tensor               enhanced_audio;
    torch::Tensor               w_tsOutAudio;
    torch::Tensor               enhanced_audio_FFT;

    //. 
    w_CAudioONNX_M.InitModel(L".\\model.onnx");

    w_nSts = w_CAudioONNX_M.OpenInputAudio(L".\\noisy.wav", w_tsInAudio, w_nTotalFrameCount, w_nSampleRate, w_nChanel);
    if (w_nSts != AD_ONNX_SUCCESS) {
        std::cout << "OpenInputAudio Error code : " << w_nSts << std::endl;
        return -1;
    }

    w_nSts = w_CAudioONNX_M.FTTranStart(w_tsInAudio, w_tsInAudio_FFT);
    if (w_nSts != AD_ONNX_SUCCESS) {
        std::cout << "FTTranStart Error code : " << w_nSts << std::endl;
        return -1;
    }
    
    // Split input audio into chunks
    std::vector<torch::Tensor> chunked_audio = w_tsInAudio_FFT.split(FRAME_SIZE);

    // Run 
    int Loop = 0;
    for (const torch::Tensor& input_frame : chunked_audio) {
        Loop++;
        w_nSts = w_CAudioONNX_M.ProcessOnnx(input_frame, w_tsOutAudio);
        if (w_nSts != AD_ONNX_SUCCESS) {
            std::cout << "ProcessOnnx Error code : " << w_nSts << std::endl;
            return -1;
        }
        else {
            std::cout << "ProcessOnnx Run Count : " << Loop << std::endl;
            output_frames.push_back(w_tsOutAudio);
        }
    }

    // Concatenate output frames
    enhanced_audio_FFT = torch::cat(output_frames).unsqueeze(0);
    
    //. 
    w_nSts = w_CAudioONNX_M.FTTranEnd(enhanced_audio_FFT, enhanced_audio);
    if (w_nSts != AD_ONNX_SUCCESS) {
        std::cout << "FTTranEnd Error code : " << w_nSts << std::endl;
        return -1;
    }

    //. Save
    w_nSts = w_CAudioONNX_M.SaveOutputAudio(L".\\cpp_output_file.wav", enhanced_audio, SF_FORMAT_WAV | SF_FORMAT_PCM_16, w_nChanel, w_nSampleRate);
    if (w_nSts != AD_ONNX_SUCCESS) {
        std::cout << "SaveOutputAudio Error code : " << w_nSts << std::endl;
        return -1;
    }

    std::cout << "Convert Audio OK." << std::endl;
    return 0;
}

