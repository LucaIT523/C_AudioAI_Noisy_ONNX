//---------------------------------------------------------------------------


#include "ad_onnx.h"



std::vector<std::string> INPUT_NAMES = { "input_frame", "states", "atten_lim_db" };
std::vector<std::string> OUTPUT_NAMES = { "enhanced_audio_frame", "out_states", "lsnr" };



std::vector<float> tensorToVector(const torch::Tensor& tensor) {
    // Get the total number of elements in the tensor
    int64_t totalElements = tensor.numel();

    // Create a vector of the appropriate size
    std::vector<float> result(totalElements);
    // Copy the tensor data to the vector
    const float* tensorData = tensor.data_ptr<float>();
    std::copy(tensorData, tensorData + totalElements, result.begin());
    return result;
}
CAudioONNX_M::CAudioONNX_M(void)
{
	m_pSession = NULL;
	m_origLen = 0;
#ifdef LD_RUN_CPU
	torch::Device torch_device(torch::kCPU);
#else
	torch::Device torch_device(torch::kCUDA);
#endif
	m_states = torch::zeros({ STATES_FULL_SIZE }, torch_device);
	m_atten_lim_db = torch::tensor(ATTEN_LIM_DB, torch_device);
	m_enhance_result = torch::zeros({ FRAME_SIZE }, torch_device);
	m_lsnr = torch::zeros({ 1 }, torch_device);
}

CAudioONNX_M::~CAudioONNX_M()
{
	if (m_pSession != NULL) {
		delete m_pSession;
	}
}

void CAudioONNX_M::InitModel(const wchar_t* p_szModelFilePath)
{
	Ort::Env env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "Default");
	Ort::SessionOptions sessionOptions;
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	sessionOptions.SetOptimizedModelFilePath(p_szModelFilePath);
	sessionOptions.SetIntraOpNumThreads(1);
	sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

	m_pSession = new Ort::Session(env, p_szModelFilePath, sessionOptions);

	return;
}

int	CAudioONNX_M::OpenInputAudio(const wchar_t* p_szInputAudioFilePath, torch::Tensor& ptsOut, int& p_nTotalFrameCnt, int& p_nSampleRate, int& p_nChanel)
{
	int			w_nRtn = AD_ERR_UNKNOWN;
	SNDFILE*	infile = NULL;
	SF_INFO		sfinfo;
	char		w_szInPath[256] = { 0, };

	if (m_pSession == NULL) {
		return AD_ONNX_ERR_INIT;
	}


	try {
		//. Read Audio Information
		SndfileHandle w_SndHandle = SndfileHandle(p_szInputAudioFilePath);
		p_nTotalFrameCnt = w_SndHandle.frames();
		p_nSampleRate = w_SndHandle.samplerate();
		p_nChanel = w_SndHandle.channels();

		std::vector<float> array(p_nTotalFrameCnt);
		sf_count_t w_ReadCnt = w_SndHandle.read(&array[0], p_nTotalFrameCnt);

		//. Convert array to tensor
		ptsOut = torch::from_blob(array.data(), array.size(), torch::kFloat32).clone();
	}
	catch (const std::exception& error) {
		return AD_ONNX_ERR_TENSOR;
	}
	//.ok
	w_nRtn = AD_ONNX_SUCCESS;
	return 	w_nRtn;
}

int	CAudioONNX_M::FTTranStart(torch::Tensor	p_tsIN, torch::Tensor&	p_tsOUT)
{
	int			w_nRtn = AD_ERR_UNKNOWN;

	if (m_pSession == NULL) {
		return AD_ONNX_ERR_INIT;
	}

	try {
		m_origLen = p_tsIN.sizes()[0];
		int hopSizeDivisiblePaddingSize = (HOP_SZIE - m_origLen % HOP_SZIE) % HOP_SZIE;
		m_origLen += hopSizeDivisiblePaddingSize;
		p_tsOUT = F::pad(p_tsIN, F::PadFuncOptions({ 0,(FFT_SIZE + hopSizeDivisiblePaddingSize) }));
	}
	catch (const std::exception& error) {
		return AD_ONNX_ERR_TENSOR;
	}

	//.ok
	w_nRtn = AD_ONNX_SUCCESS;
	return 	w_nRtn;
}
int	CAudioONNX_M::FTTranEnd(torch::Tensor	p_tsIN, torch::Tensor& p_tsOUT)
{
	int			w_nRtn = AD_ERR_UNKNOWN;

	if (m_pSession == NULL) {
		return AD_ONNX_ERR_INIT;
	}
	if (m_origLen <= 0) {
		return AD_ONNX_ERR_FFT_START;
	}

	try {
		int pos = FFT_SIZE - HOP_SZIE;
		p_tsOUT = p_tsIN.slice(1, pos, m_origLen + pos);
	}
	catch (const std::exception& error) {
		return AD_ONNX_ERR_TENSOR;
	}

	//.ok
	w_nRtn = AD_ONNX_SUCCESS;
	return 	w_nRtn;
}
int	CAudioONNX_M::ProcessOnnx(torch::Tensor	p_tsIN, torch::Tensor& p_tsOUT)
{
	int						w_nRtn = AD_ERR_UNKNOWN;
	std::vector<Ort::Value> ort_inputs;
	std::vector<Ort::Value> ort_outputs;
	std::vector<const char*> input_names;
	std::vector<const char*> output_names;

	if (m_pSession == NULL) {
		return AD_ONNX_ERR_INIT;
	}
	for (size_t i = 0; i < INPUT_NAMES.size(); i++) {
		input_names.emplace_back(INPUT_NAMES[i].c_str());
		output_names.emplace_back(OUTPUT_NAMES[i].c_str());
	}
	try {
		//. Ort inputs
		Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, p_tsIN.data_ptr<float>(), p_tsIN.numel(), p_tsIN.sizes().data(), p_tsIN.sizes().size()));
		ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, m_states.data_ptr<float>(), m_states.numel(), m_states.sizes().data(), m_states.sizes().size()));
		ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, m_atten_lim_db.data_ptr<float>(), m_atten_lim_db.numel(), m_atten_lim_db.sizes().data(), m_atten_lim_db.sizes().size()));
		//. Ort outputs
		ort_outputs.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, m_enhance_result.data_ptr<float>(), m_enhance_result.numel(), m_enhance_result.sizes().data(), m_enhance_result.sizes().size()));
		ort_outputs.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, m_states.data_ptr<float>(), m_states.numel(), m_states.sizes().data(), m_states.sizes().size()));
		ort_outputs.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, m_lsnr.data_ptr<float>(), m_lsnr.numel(), m_lsnr.sizes().data(), m_lsnr.sizes().size()));

		m_pSession->Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(), output_names.data(), ort_outputs.data(), ort_outputs.size());

		Ort::Value& outputTensorOrt = ort_outputs[0];
		const Ort::TensorTypeAndShapeInfo& outputInfo = outputTensorOrt.GetTensorTypeAndShapeInfo();
		p_tsOUT = torch::from_blob(outputTensorOrt.GetTensorMutableData<float>(), outputInfo.GetShape(), torch::kFloat32).clone();
	}
	catch (const std::exception& error) {
		return AD_ONNX_ERR_TENSOR;
	}

	//.ok
	w_nRtn = AD_ONNX_SUCCESS;
	return 	w_nRtn;
}
int	 CAudioONNX_M::SaveOutputAudio(const wchar_t* p_szOutputAudioFilePath, torch::Tensor	p_tsIN, int p_Formt, int p_Ch, int p_SRate)
{
	int				w_nRtn = AD_ERR_UNKNOWN;
	SndfileHandle	file;

	if (m_pSession == NULL) {
		return AD_ONNX_ERR_INIT;
	}
	try {

		std::vector<float> audio_data = tensorToVector(p_tsIN);
		file = SndfileHandle(p_szOutputAudioFilePath, SFM_WRITE, p_Formt, p_Ch, p_SRate);
		file.write(&audio_data[0], audio_data.size());
	}
	catch (const std::exception& error) {
		return AD_ONNX_ERR_FILE;
	}
	//.ok
	w_nRtn = AD_ONNX_SUCCESS;
	return 	w_nRtn;
}



