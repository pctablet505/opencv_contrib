// #30919: com.microsoft Attention with the spec's int32 mask_index [B, S] (graph input, constant), and without a bias
#include <opencv2/dnn.hpp>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

static cv::Mat readRaw(const std::string& path, const std::vector<int>& shape, int type)
{
    cv::Mat m(shape, type);
    std::ifstream(path, std::ios::binary).read((char*)m.data, m.total() * m.elemSize());
    return m;
}

int main(int argc, char** argv)
{
    const std::string dir = argv[1];
    cv::Mat x = readRaw(dir + "/attention_mask_x.bin", {1, 16, 64}, CV_32F);
    cv::Mat m = readRaw(dir + "/attention_mask_m.bin", {1, 16}, CV_32S);
    cv::Mat ref = readRaw(dir + "/attention_mask_ref.bin", {1, 16, 64}, CV_32F);
    const struct { const char* name; int backend, target; } runs[] = {
        {"CPU", cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU}, {"CUDA", cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA}};
    for (const char* model : {"attention_mask_input", "attention_mask_const", "attention_nobias", "attention_emptybias"})
        for (const auto& r : runs)
        {
            try
            {
                cv::dnn::Net net = cv::dnn::readNetFromONNX(dir + "/" + model + ".onnx");
                net.setPreferableBackend(r.backend);
                net.setPreferableTarget(r.target);
                if (std::string(model) == "attention_mask_input")
                {
                    net.setInput(x, "X");
                    net.setInput(m, "M");
                }
                else
                    net.setInput(x);
                cv::Mat y = net.forward();
                if (std::string(model).rfind("attention_mask", 0) == 0)
                    std::printf("%s %s: max |OpenCV - onnxruntime| = %g\n", model, r.name, cv::norm(y.reshape(1, 1), ref.reshape(1, 1), cv::NORM_INF));
                else
                    std::printf("%s %s: runs\n", model, r.name);
            }
            catch (const cv::Exception& e)
            {
                std::printf("%s %s: %s\n", model, r.name, e.what());
            }
        }
    return 0;
}
