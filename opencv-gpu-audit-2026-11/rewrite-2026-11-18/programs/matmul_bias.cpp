// #30922: CUDA MatMul with a constant 3-D B: alpha, and the device memory taken by a broadcast bias
#include <opencv2/dnn.hpp>
#include <opencv2/core/cuda.hpp>
#include <cstdio>

using namespace cv;
using namespace cv::dnn;

static Mat run(LayerParams lp, const Mat& a, int backend, int target)
{
    Net net;
    net.addLayerToPrev(lp.name, lp.type, lp);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    net.setInput(a);
    return net.forward().clone();
}

int main()
{
    // 1. alpha = 1.5, beta = 0.5: CPU and CUDA should agree
    {
        Mat A({2, 5, 12}, CV_32F), B({1, 12, 16}, CV_32F), bias(std::vector<int>{16}, CV_32F);
        randu(A, -1.f, 1.f); randu(B, -0.5f, 0.5f); randu(bias, 1.f, 2.f);
        LayerParams lp;
        lp.type = "MatMul"; lp.name = "mm";
        lp.set("alpha", 1.5f); lp.set("beta", 0.5f); lp.set("real_ndims_C", 1);
        lp.blobs = {B, bias};
        Mat cpu = run(lp, A, DNN_BACKEND_OPENCV, DNN_TARGET_CPU);
        Mat gpu = run(lp, A, DNN_BACKEND_CUDA, DNN_TARGET_CUDA);
        std::printf("alpha 1.5, beta 0.5: max |CPU - CUDA| = %g (expected < 1e-5)\n", norm(cpu, gpu, NORM_INF));
    }
    // 2. [16, 1024, 128] x [1, 128, 1024] + bias[1024]: device memory kept after the first forward()
    {
        Mat A({16, 1024, 128}, CV_32F), B({1, 128, 1024}, CV_32F), bias(std::vector<int>{1024}, CV_32F);
        randu(A, -1.f, 1.f); randu(B, -0.1f, 0.1f); randu(bias, -1.f, 1.f);
        LayerParams lp;
        lp.type = "MatMul"; lp.name = "mm";
        lp.set("real_ndims_C", 1);
        lp.blobs = {B, bias};
        run(lp, A, DNN_BACKEND_CUDA, DNN_TARGET_CUDA);  // loads the CUDA libraries
        cuda::DeviceInfo info;
        const size_t free0 = info.freeMemory();
        Net net;
        net.addLayerToPrev(lp.name, lp.type, lp);
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA);
        net.setInput(A);
        net.forward();
        const size_t free1 = info.freeMemory();
        std::printf("[16, 1024, 128] x [1, 128, 1024] + bias[1024]: %.0f MiB of device memory after the first forward()\n",
                    (double)(free0 - free1) / (1 << 20));
        std::printf("(input 8 MiB, weight 0.5 MiB, output 64 MiB)\n");
    }
    return 0;
}
