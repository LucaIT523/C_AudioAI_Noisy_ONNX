#ifndef __AD_ONNX_InferenceH
#define __AD_ONNX_InferenceH

#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <sndfile.hh>
#include "torch/script.h"
#include "torch/torch.h"

using namespace std;
using namespace Ort;
namespace F = torch::nn::functional;

//---------------------------------------------------------------------------
constexpr int FRAME_SIZE = 480;
constexpr int STATES_FULL_SIZE = 45304;
constexpr float ATTEN_LIM_DB = 0.0;
constexpr int HOP_SZIE = 480;
constexpr int FFT_SIZE = 960;
constexpr int SAMPLE_RATE = 48000;



#define		LD_RUN_CPU


//. Error Code
#define		AD_ONNX_SUCCESS			0
#define		AD_ONNX_ERR_INIT		-1
#define		AD_ONNX_ERR_FILE		-2
#define		AD_ONNX_ERR_TENSOR		-3
#define		AD_ONNX_ERR_FFT_START	-4

#define		AD_ERR_UNKNOWN			-99

class CAudioONNX_M
{
public:

	CAudioONNX_M();
	~CAudioONNX_M();

	void			InitModel(const wchar_t* p_szModelFilePath);

	int				OpenInputAudio(const wchar_t* p_szInputAudioFilePath, torch::Tensor&	ptsOut, int&	p_nTotalFrameCnt, int& p_nSampleRate, int&	p_nChanel);

	int				FTTranStart(torch::Tensor	p_tsIN, torch::Tensor&	p_tsOUT);

	int				ProcessOnnx(torch::Tensor	p_tsIN, torch::Tensor&	p_tsOUT);

	int				FTTranEnd(torch::Tensor	p_tsIN, torch::Tensor&	p_tsOUT);

	int				SaveOutputAudio(const wchar_t* p_szOutputAudioFilePath, torch::Tensor	p_tsIN, int p_Formt, int p_Ch, int p_SRate);

private:
	Ort::Session*	m_pSession;

	torch::Tensor	m_states;
	torch::Tensor	m_atten_lim_db ;
	torch::Tensor	m_enhance_result;
	torch::Tensor	m_lsnr;

	int				m_origLen;

};





#endif // __AD_ONNX_InferenceH