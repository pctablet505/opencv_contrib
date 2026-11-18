// Device memory taken by the first forward() of a com.microsoft Attention node, and the forward() time.
// Usage: attention_sweep <attention_S<S>.onnx> <S> <fp32|fp16>
#include <opencv2/dnn.hpp>
#include <opencv2/core/cuda.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
using namespace cv;
using namespace cv::dnn;

int main(int argc, char** argv)
{
    const int S = std::atoi(argv[2]);
    const bool fp16 = std::string(argv[3]) == "fp16";
    cuda::DeviceInfo info;
    Mat x(std::vector<int>{1, S, 768}, CV_32F);
    randu(x, -1, 1);
    {
        // a first Net loads the CUDA libraries, so the delta below counts only what this node keeps
        Net warm = readNetFromONNX(argv[1]);
        warm.setPreferableBackend(DNN_BACKEND_CUDA);
        warm.setPreferableTarget(fp16 ? DNN_TARGET_CUDA_FP16 : DNN_TARGET_CUDA);
        warm.setInput(x);
        warm.forward();
    }
    Net net = readNetFromONNX(argv[1]);
    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(fp16 ? DNN_TARGET_CUDA_FP16 : DNN_TARGET_CUDA);
    const size_t free0 = info.freeMemory();
    net.setInput(x);
    Mat y = net.forward().clone();
    const size_t free1 = info.freeMemory();
    for (int i = 0; i < 10; i++) net.forward();
    std::vector<double> ms;
    for (int i = 0; i < 50; i++)
    {
        auto t0 = std::chrono::steady_clock::now();
        net.forward();
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    std::printf("{\"S\":%d,\"precision\":\"%s\",\"device_mib\":%.1f,\"median_ms\":%.4f,\"p10_ms\":%.4f,\"p90_ms\":%.4f,\"checksum\":%.6g}\n",
                S, fp16 ? "fp16" : "fp32", (double)(free0 - free1) / (1 << 20), ms[25], ms[5], ms[45], sum(y)[0]);
    return 0;
}
