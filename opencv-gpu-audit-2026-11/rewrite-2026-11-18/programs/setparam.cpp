// #30918: Net::setParam() on the CUDA backend, classic Net and ONNX graph
#include <opencv2/dnn.hpp>
#include <cstdio>

int main(int argc, char** argv)
{
    cv::Mat w2 = cv::Mat::eye(4, 4, CV_32F) * 2.f, w5 = cv::Mat::eye(4, 4, CV_32F) * 5.f;
    cv::Mat x = cv::Mat::ones(1, 4, CV_32F);

    // classic Net
    {
        cv::dnn::LayerParams lp;
        lp.type = "Gemm"; lp.name = "fc";
        lp.set("constB", true);
        lp.blobs.push_back(w2.clone());
        cv::dnn::Net net;
        int id = net.addLayerToPrev(lp.name, lp.type, lp);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        net.setInput(x);
        std::printf("classic Net, weight 2*I: %g\n", net.forward().at<float>(0, 0));
        net.setParam(id, 0, w5);
        std::printf("classic Net, after setParam(5*I): %g (expected 5)\n", net.forward().at<float>(0, 0));
    }

    // ONNX graph (Y = Gemm(X, W), W = 2*I): the second forward records a CUDA Graph, later ones replay it
    {
        cv::dnn::Net net = cv::dnn::readNetFromONNX(argv[1]);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        net.setInput(x);
        for (int call = 1; call <= 3; call++)
            std::printf("ONNX graph, call %d: %g\n", call, net.forward().at<float>(0, 0));
        net.setParam("Y", 0, w5);
        for (int call = 4; call <= 5; call++)
            std::printf("ONNX graph, call %d after setParam(5*I): %g (expected 5)\n", call, net.forward().at<float>(0, 0));
    }

    // ONNX graph, setParam() between the first and the second forward
    {
        cv::dnn::Net net = cv::dnn::readNetFromONNX(argv[1]);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        net.setInput(x);
        std::printf("ONNX graph, call 1: %g\n", net.forward().at<float>(0, 0));
        net.setParam("Y", 0, w5);
        try
        {
            std::printf("ONNX graph, call 2 after setParam(5*I): %g (expected 5)\n", net.forward().at<float>(0, 0));
            std::printf("ONNX graph, call 3: %g (expected 5)\n", net.forward().at<float>(0, 0));
        }
        catch (const cv::Exception& e)
        {
            std::printf("ONNX graph, call 2 threw: %s\n", e.what());
        }
    }
    return 0;
}
