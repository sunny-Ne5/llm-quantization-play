#include<quantize.h>

std::tuple<torch::Tensor, float, int> QuantizeCore::quantizeTensor(const torch::Tensor& tensor)
{
    switch (m_qType)
    {
        case Q_4_BIT:
            return Q_4_Core().quantizeTensor(tensor);
        
        case Q_8_BIT:
            return Q_8_Core().quantize(tensor);
        
        default:
            throw std::runtime_error("something went wrong");
    }
}

std::tuple<torch::Tensor, float, int> Q_8_Core::quantizeTensor(const torch::Tensor& tensor)
{
    // Find min and max values
    float min_val = torch::min(tensor).item<float>();
    float max_val = torch::max(tensor).item<float>();

    // Calculate scale and zero point
    float scale = (max_val - min_val) / 255.0f;
    int zero_point = static_cast<int>(-min_val / scale) - 128;

    // Quantize the tensor
    torch::Tensor quantized_tensor = torch::round((tensor - min_val) / scale - 128).to(torch::kInt8);

    return std::make_tuple(quantized_tensor, scale, zero_point);
}

std::tuple<torch::Tensor, float, int> Q_8_Core::quantizeTensor(const torch::Tensor& tensor)
{
    // TODO: Implement
    return {};
}