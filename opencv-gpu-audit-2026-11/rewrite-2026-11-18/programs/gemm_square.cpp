// #30915: a square Gemm with transB = 0 on the CUDA backend
#include <opencv2/dnn.hpp>
#include <iostream>

int main()
{
    // upper-triangular W, so X * W and X * W^T are easy to tell apart
    float w[] = {1, 2, 3, 4,
                 0, 1, 2, 3,
                 0, 0, 1, 2,
                 0, 0, 0, 1};
    cv::Mat W(4, 4, CV_32F, w);
    cv::dnn::LayerParams lp;
    lp.type = "Gemm"; lp.name = "gemm";
    lp.set("transB", false); lp.set("constB", true);
    lp.blobs.push_back(W.clone());

    cv::dnn::Net net;
    int id = net.addLayerToPrev(lp.name, lp.type, lp);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    cv::Mat x1 = cv::Mat::ones(1, 4, CV_32F), x2 = cv::Mat::ones(2, 4, CV_32F);
    std::cout << "expected X * W:       " << x1 * W << "\n";
    net.setInput(x1);
    std::cout << "batch 1:              " << net.forward().row(0) << "\n";
    std::cout << "weight row 0 after:   " << net.getParam(id, 0).row(0) << " (was [1, 2, 3, 4])\n";
    net.setInput(x2);
    std::cout << "batch 2, row 0:       " << net.forward().row(0) << "\n";
}
