// CPU normalization layers against an FP64 reference.
//   sweep: max output error as the input mean grows (values are mean + U(-0.5, 0.5), so std = 0.2887)
//   hash:  a hash of the outputs for ordinary inputs, to compare two builds bit for bit
//   long:  max MVN error on one long row at a mean of 10000
//   time:  single-thread forward() time on ordinary rows and on rows at a mean of 10000
// Usage: norm_offset_cpu sweep | hash | long | time
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
using namespace cv;
using namespace cv::dnn;

static const int C = 64, W = 768, GROUPS = 32;

// x is {1, C', 1, W'}. LayerNormalization, MVN and InstanceNormalization normalize each channel's W' values;
// GroupNormalization each group of C' / GROUPS channels.
static Net makeNet(const std::string& type, const Mat& x)
{
    const int c = x.size[1], w = (int)(x.total() / c);
    LayerParams lp;
    lp.type = type;
    lp.name = "norm";
    std::vector<Mat> inputs;
    if (type == "LayerNormalization")
    {
        lp.set("axis", 2);
        lp.set("epsilon", 1e-5f);
        inputs = {x.reshape(1, std::vector<int>{1, c, w}), Mat(std::vector<int>{w}, CV_32F, Scalar(1)),
                  Mat(std::vector<int>{w}, CV_32F, Scalar(0))};
    }
    else if (type == "MVN")
    {
        lp.set("normalize_variance", true);
        lp.set("eps", 1e-5f);
        inputs = {x};
    }
    else
    {
        lp.set("epsilon", 1e-5f);
        if (type == "GroupNormalization")
            lp.set("num_groups", GROUPS);
        inputs = {x, Mat(std::vector<int>{c}, CV_32F, Scalar(1)), Mat(std::vector<int>{c}, CV_32F, Scalar(0))};
    }
    Net net;
    const int id = net.addLayer(lp.name, lp.type, lp);
    std::vector<String> names;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        net.connect(0, (int)i, id, (int)i);
        names.push_back(format("in%d", (int)i));
    }
    net.setInputsNames(names);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
    for (size_t i = 0; i < inputs.size(); i++)
        net.setInput(inputs[i], names[i]);
    return net;
}

// Largest |y - FP64 reference| over rows of len values.
static double maxError(const Mat& x, const Mat& y, int len)
{
    const float* xp = x.ptr<float>();
    const float* yp = y.ptr<float>();
    double maxErr = 0;
    for (size_t r = 0; r < x.total() / len; r++)
    {
        const float* xr = xp + r * len;
        double s = 0, q = 0;
        for (int j = 0; j < len; j++) s += xr[j];
        const double mean = s / len;
        for (int j = 0; j < len; j++) q += (xr[j] - mean) * (xr[j] - mean);
        const double inv = 1. / std::sqrt(q / len + 1e-5);
        for (int j = 0; j < len; j++)
        {
            const float v = yp[r * len + j];
            maxErr = std::max(maxErr, std::isnan(v) ? INFINITY : std::abs(v - (xr[j] - mean) * inv));
        }
    }
    return maxErr;
}

static Mat offsetInput(const std::vector<int>& shape, double mean)
{
    Mat x(shape, CV_32F);
    RNG rng(12345);
    rng.fill(x, RNG::UNIFORM, -0.5, 0.5);
    x += Scalar(mean);
    return x;
}

int main(int argc, char** argv)
{
    const std::string mode = argc > 1 ? argv[1] : "sweep";
    const char* layers[] = {"LayerNormalization", "MVN", "InstanceNormalization", "GroupNormalization"};
    if (mode == "hash")
    {
        for (const char* type : layers)
            for (int input = 0; input < 3; input++)
            {
                Mat x(std::vector<int>{1, C, 1, W}, CV_32F);
                RNG rng(777 + input);
                if (input == 0) rng.fill(x, RNG::NORMAL, 0, 1);       // mean 0
                if (input == 1) rng.fill(x, RNG::UNIFORM, 0, 1);      // mean 1.7 std
                if (input == 2) rng.fill(x, RNG::NORMAL, 5, 1);       // mean 5 std
                Mat y = makeNet(type, x).forward();
                uint64_t h = 1469598103934665603ull;                   // FNV-1a over the output bytes
                for (size_t i = 0; i < y.total() * y.elemSize(); i++)
                    h = (h ^ y.data[i]) * 1099511628211ull;
                std::printf("{\"layer\":\"%s\",\"input\":%d,\"hash\":\"%016llx\"}\n", type, input, (unsigned long long)h);
            }
        return 0;
    }
    if (mode == "long")
    {
        for (int side : {512, 1024, 2048})
        {
            Mat x = offsetInput({1, 1, side, side}, 10000);
            Mat y = makeNet("MVN", x).forward();
            std::printf("{\"layer\":\"MVN\",\"row\":%d,\"mean\":10000,\"max_err\":%.6g}\n", side * side, maxError(x, y, side * side));
        }
        return 0;
    }
    if (mode == "time")
    {
        setNumThreads(1);
        const struct { const char* type; std::vector<int> shape; } cases[] = {
            {"LayerNormalization", {1, 512, 1, 768}}, {"MVN", {1, 64, 180, 240}}};
        for (const auto& c : cases)
            for (double mean : {0.5, 10000.})
            {
                Mat x = offsetInput(c.shape, mean);
                Net net = makeNet(c.type, x);
                for (int i = 0; i < 5; i++) net.forward();
                std::vector<double> us;
                for (int i = 0; i < 50; i++)
                {
                    auto t0 = std::chrono::steady_clock::now();
                    net.forward();
                    us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
                }
                std::sort(us.begin(), us.end());
                std::printf("{\"layer\":\"%s\",\"mean\":%g,\"median_us\":%.1f,\"p10_us\":%.1f,\"p90_us\":%.1f}\n",
                            c.type, mean, us[25], us[5], us[45]);
            }
        return 0;
    }
    const double offsets[] = {1, 10, 100, 300, 1000, 2000, 3000, 5000, 10000, 30000, 100000};
    for (const char* type : layers)
        for (double c : offsets)
        {
            Mat x = offsetInput({1, C, 1, W}, c);
            Mat y = makeNet(type, x).forward();
            const int len = std::string(type) == "GroupNormalization" ? W * C / GROUPS : W;
            int nans = 0;
            for (size_t i = 0; i < y.total(); i++) nans += std::isnan(y.ptr<float>()[i]);
            std::printf("{\"layer\":\"%s\",\"mean\":%g,\"nan\":%d,\"max_err\":%.6g}\n", type, c, nans, maxError(x, y, len));
        }
    return 0;
}
