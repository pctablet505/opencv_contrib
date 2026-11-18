// #30920: MatMul + Add fused into Gemm, with a [1, 1, N] bias and with an [M, 1] column bias
#include <opencv2/dnn.hpp>
#include <iostream>

int main(int argc, char** argv)
{
    const std::string dir = argv[1];
    const int backend = argc > 2 && std::string(argv[2]) == "cuda" ? cv::dnn::DNN_BACKEND_CUDA : cv::dnn::DNN_BACKEND_OPENCV;
    const int target = backend == cv::dnn::DNN_BACKEND_CUDA ? cv::dnn::DNN_TARGET_CUDA : cv::dnn::DNN_TARGET_CPU;

    // Y = X[2, 4, 8] * W[8, 6] + B[1, 1, 6]
    try
    {
        cv::dnn::Net net = cv::dnn::readNetFromONNX(dir + "/matmul_bias_1x1xN.onnx");
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
        net.setInput(cv::Mat(std::vector<int>{2, 4, 8}, CV_32F, cv::Scalar(0)));
        cv::Mat y = net.forward();
        std::cout << "[1, 1, N] bias, X = 0: Y[0, 0, :] = " << cv::Mat(1, 6, CV_32F, y.ptr<float>()) << " (expected [1, 2, 3, 4, 5, 6])\n";
    }
    catch (const cv::Exception& e)
    {
        std::cout << "[1, 1, N] bias: " << e.what() << "\n";
    }

    // Y = X[3, 4] * W[4, 3] + B[3, 1]; with X = 0, Y is the bias broadcast along each row
    cv::dnn::Net net = cv::dnn::readNetFromONNX(dir + "/matmul_bias_Mx1.onnx");
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    net.setInput(cv::Mat::zeros(3, 4, CV_32F));
    std::cout << "[M, 1] bias [1; 2; 3], X = 0:\n" << net.forward() << "\nexpected:\n[1, 1, 1;\n 2, 2, 2;\n 3, 3, 3]\n";
    return 0;
}
