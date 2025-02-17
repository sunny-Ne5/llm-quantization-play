#include <torch/extension.h>
#include<vector>

enum QuantizationType
{
    Q_4_BIT =0,
    Q_8_BIT
};  

class QuantizeCore
{
public:
    QuantizeCore(QuantizationType qType) : m_qType(qType) {}

    // Quantize tensor, scale and zero point
    std::tuple<torch::Tensor, float, int> quantizeTensor(const torch::Tensor& tensor);    

private:
    QuantizationType m_qType;
};

4 bit quantization implementation
class Q_4_Core
{
public:
    std::tuple<torch::Tensor, float, int> quantizeTensor(const torch::Tensor& tensor);
};

/* 8 bit quantization implementation*/
class Q_8_Core
{
public:
    std::tuple<torch::Tensor, float, int> quantizeTensor(const torch::Tensor& tensor);
};


