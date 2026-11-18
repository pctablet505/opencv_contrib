// Output error of the CUDA and CPU normalization layers against an FP64 reference, as the input mean grows.
// Usage: norm_offset sweep | norm_offset time <layernorm_x24.onnx>
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
using namespace cv;
using namespace cv::dnn;

static Mat runNorm(const std::string& type, const Mat& x3, int backend, int target)
{
    const int rows = x3.size[1], inner = x3.size[2];
    LayerParams lp;
    lp.type = type;
    lp.name = "norm";
    Net net;
    if (type == "LayerNormalization")
    {
        lp.set("axis", 2);
        lp.set("epsilon", 1e-5f);
        int id = net.addLayer(lp.name, lp.type, lp);
        net.connect(0, 0, id, 0);
        net.connect(0, 1, id, 1);
        net.connect(0, 2, id, 2);
        net.setInputsNames({"X", "scale", "bias"});
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
        Mat scale(std::vector<int>{inner}, CV_32F, Scalar(1)), bias(std::vector<int>{inner}, CV_32F, Scalar(0));
        net.setInput(x3, "X");
        net.setInput(scale, "scale");
        net.setInput(bias, "bias");
    }
    else
    {
        lp.set("normalize_variance", true);
        lp.set("eps", 1e-5f);
        net.addLayerToPrev(lp.name, lp.type, lp);
        net.setPreferableBackend(backend);
        net.setPreferableTarget(target);
        net.setInput(x3.reshape(1, std::vector<int>{1, rows, 1, inner}));
    }
    return net.forward().clone();
}

int main(int argc, char** argv)
{
    const std::string mode = argc > 1 ? argv[1] : "sweep";
    if (mode == "time")
    {
        Net net = readNetFromONNX(argv[2]);
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA);
        Mat x(std::vector<int>{1, 1024, 768}, CV_32F);
        randu(x, -1, 1);
        net.setInput(x);
        for (int i = 0; i < 20; i++) net.forward();
        std::vector<double> ms;
        for (int i = 0; i < 100; i++)
        {
            auto t0 = std::chrono::steady_clock::now();
            net.forward();
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(ms.begin(), ms.end());
        std::printf("{\"mode\":\"time\",\"model\":\"layernorm_x24 [1,1024,768]\",\"median_ms\":%.4f,\"p10_ms\":%.4f,\"p90_ms\":%.4f}\n",
                    ms[50], ms[10], ms[90]);
        return 0;
    }

    const int rows = 64, inner = 768;
    const double offsets[] = {1, 10, 100, 300, 1000, 2000, 3000, 5000, 10000, 30000, 100000};
    const struct { const char* name; int backend, target; } runs[] = {
        {"CUDA", DNN_BACKEND_CUDA, DNN_TARGET_CUDA}, {"CPU", DNN_BACKEND_OPENCV, DNN_TARGET_CPU}};
    for (const char* type : {"LayerNormalization", "MVN"})
        for (const auto& r : runs)
            for (double c : offsets)
            {
                Mat x(std::vector<int>{1, rows, inner}, CV_32F);
                RNG rng(12345);
                rng.fill(x, RNG::UNIFORM, -0.5, 0.5);
                x += Scalar(c);
                Mat y = runNorm(type, x, r.backend, r.target);
                const float* xp = x.ptr<float>();
                const float* yp = y.ptr<float>();
                int nans = 0;
                double maxErr = 0, sd = 0;
                for (int i = 0; i < rows; i++)
                {
                    double s = 0, q = 0;
                    for (int j = 0; j < inner; j++) s += xp[i * inner + j];
                    const double mean = s / inner;
                    for (int j = 0; j < inner; j++) q += (xp[i * inner + j] - mean) * (xp[i * inner + j] - mean);
                    const double var = q / inner;
                    sd += std::sqrt(var) / rows;
                    for (int j = 0; j < inner; j++)
                    {
                        const double ref = (xp[i * inner + j] - mean) / std::sqrt(var + 1e-5);
                        const float v = yp[i * inner + j];
                        if (std::isnan(v)) nans++;
                        else maxErr = std::max(maxErr, std::abs(v - ref));
                    }
                }
                std::printf("{\"layer\":\"%s\",\"backend\":\"%s\",\"mean\":%g,\"std\":%.4f,\"nan\":%d,\"total\":%d,\"max_err\":%.6g}\n",
                            type, r.name, c, sd, nans, rows * inner, maxErr);
            }
    return 0;
}
