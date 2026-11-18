// #30916: LayerNormalization on rows whose mean is large compared with their spread, CUDA and CPU
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>

static cv::Mat layerNorm(const cv::Mat& x, int backend, int target)
{
    cv::dnn::LayerParams lp;
    lp.type = "LayerNormalization"; lp.name = "ln";
    lp.set("axis", 2); lp.set("epsilon", 1e-5f);
    cv::dnn::Net net;
    int id = net.addLayer(lp.name, lp.type, lp);
    net.connect(0, 0, id, 0); net.connect(0, 1, id, 1); net.connect(0, 2, id, 2);
    net.setInputsNames({"X", "scale", "bias"});
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    const int n = x.size[2];
    net.setInput(x, "X");
    net.setInput(cv::Mat(std::vector<int>{n}, CV_32F, cv::Scalar(1)), "scale");
    net.setInput(cv::Mat(std::vector<int>{n}, CV_32F, cv::Scalar(0)), "bias");
    return net.forward().clone();
}

int main()
{
    // 8 rows of 768 values: 10000 plus a uniform spread of +-0.5 (standard deviation 0.29)
    const int rows = 8, n = 768;
    cv::Mat x(std::vector<int>{1, rows, n}, CV_32F);
    cv::randu(x, -0.5f, 0.5f);
    x += cv::Scalar(10000);
    const float* xp = x.ptr<float>();

    for (auto [name, backend, target] : {std::make_tuple("CUDA", cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA),
                                         std::make_tuple("CPU ", cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU)})
    {
        cv::Mat y = layerNorm(x, backend, target);
        const float* yp = y.ptr<float>();
        int nans = 0;
        double maxErr = 0;
        for (int r = 0; r < rows; r++)
        {
            double mean = 0, var = 0;
            for (int i = 0; i < n; i++) mean += xp[r * n + i];
            mean /= n;
            for (int i = 0; i < n; i++) var += (xp[r * n + i] - mean) * (xp[r * n + i] - mean);
            var /= n;
            for (int i = 0; i < n; i++)
            {
                const double ref = (xp[r * n + i] - mean) / std::sqrt(var + 1e-5);
                if (std::isnan(yp[r * n + i])) nans++;
                else maxErr = std::max(maxErr, std::abs(yp[r * n + i] - ref));
            }
        }
        std::printf("%s: %d of %d outputs NaN, largest error of the others %.3g (outputs should lie within +-1.73)\n",
                    name, nans, rows * n, maxErr);
    }
    return 0;
}
