// #30917: CUDA UMat allocator (cv::cuda::getCudaAllocator()) on 5.x
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <cstdio>
#include <vector>

int main()
{
    cv::MatAllocator* a = cv::cuda::getCudaAllocator();

    // 1. A UMat over the caller's host buffer: reads return device memory that was never written
    std::vector<float> host(64, 42.5f);
    int sz[] = {8, 8};
    size_t st[] = {8 * sizeof(float), sizeof(float)};
    cv::UMatData* u = a->allocate(2, sz, CV_32F, host.data(), st, cv::ACCESS_RW, cv::USAGE_DEFAULT);
    float v = -1.f;
    size_t region[] = {1, sizeof(float)}, srcofs[] = {0, 0}, srcstep[] = {8 * sizeof(float)}, dststep[] = {sizeof(float)};
    a->download(u, &v, 2, region, srcofs, srcstep, dststep);
    std::printf("1. first element read back: %g (expected 42.5)\n", v);
    a->deallocate(u);

    // 2. A 0-D allocation writes step[-1]
    size_t steps[3] = {111, 222, 333};
    cv::UMatData* u0 = a->allocate(0, NULL, CV_32F, NULL, &steps[1], cv::ACCESS_RW, cv::USAGE_DEFAULT);
    std::printf("2. word before the step array: %zu (expected 111)\n", steps[0]);
    a->deallocate(u0);

    // 3. Taking over a UMatData from another allocator leaves currAllocator unchanged, so the device buffer is never freed
    int big[] = {1024, 1024};
    size_t bst[] = {CV_AUTOSTEP, CV_AUTOSTEP};
    size_t free0 = cv::cuda::DeviceInfo().freeMemory();
    for (int i = 0; i < 100; i++)
    {
        cv::UMatData* h = cv::Mat::getStdAllocator()->allocate(2, big, CV_32F, NULL, bst, cv::ACCESS_RW, cv::USAGE_DEFAULT);
        a->allocate(h, cv::ACCESS_RW, cv::USAGE_DEFAULT);
        if (i == 0)
            std::printf("3. owner is the CUDA allocator after the take-over: %d (expected 1)\n", h->currAllocator == a);
        h->currAllocator->deallocate(h);
    }
    size_t free1 = cv::cuda::DeviceInfo().freeMemory();
    std::printf("   device memory lost after 100 take-overs of a 4 MiB buffer: %.0f MiB (expected 0)\n",
                (double)(free0 - free1) / (1 << 20));
    return 0;
}
