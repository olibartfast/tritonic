#pragma once
#include "TaskInterface.hpp"
#include <algorithm>
#include <numeric>

class TorchvisionClassifier : public TaskInterface {
public:
    TorchvisionClassifier(int input_width, int input_height, int channels)
        : TaskInterface(input_width, input_height, channels) {
    }

    std::vector<Result> postprocess(const cv::Size& frame_size, std::vector<std::vector<float>>& infer_results,
                                            const std::vector<std::vector<int64_t>>& infer_shapes) override 
    {
        
        // Implement your postprocess logic here
        auto result = infer_results.front();
        const auto shape = infer_shapes[0][1];
        std::transform(result.begin(), result.end(), result.begin(), [](float val) {return std::exp(val);});
        auto sum = std::accumulate(result.begin(), result.end(), 0.0);
        // find top classes predicted by the model
        std::vector<int> indices(shape);
        std::iota(indices.begin(), indices.end(), 0); // generate sequence 0, 1, 2, 3, ..., 999
        std::sort(indices.begin(), indices.end(), [&result](int i1, int i2) {return result[i1] > result[i2];});
        // print results
        int i = 0;
        std::vector<Result> results;
        while (result[indices[i]] / sum > 0.005)
        {
            Classification classification;
            classification.class_id = indices[i];
            classification.class_confidence = result[indices[i]] / sum;
            ++i;
            results.emplace_back(classification);
        }
            return results;
    }

    std::vector<uint8_t> preprocess(const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
                                    size_t img_channels, const cv::Size& img_size) override {
    std::vector<uint8_t> input_data;
        cv::Mat sample;
        cv::cvtColor(img, sample,  cv::COLOR_BGR2RGB);
        sample.convertTo(
            sample, (img_channels == 3) ? img_type3 : img_type1);
        sample.convertTo(sample, CV_32FC3, 1.f / 255.f);
        cv::subtract(sample, cv::Scalar(0.485f, 0.456f, 0.406f), sample, cv::noArray(), -1);
        cv::divide(sample, cv::Scalar(0.229f, 0.224f, 0.225f), sample, 1, -1);
        cv::resize(sample, sample, cv::Size(img_size.width, img_size.height));

        // Allocate a buffer to hold all image elements.
        size_t img_byte_size = sample.total() * sample.elemSize();
        size_t pos = 0;
        input_data.resize(img_byte_size);

        // (format.compare("FORMAT_NCHW") == 0)
        //
        // For CHW formats must split out each channel from the matrix and
        // order them as BBBB...GGGG...RRRR. To do this split the channels
        // of the image directly into 'input_data'. The BGR channels are
        // backed by the 'input_data' vector so that ends up with CHW
        // order of the data.
        std::vector<cv::Mat> input_bgr_channels;
        for (size_t i = 0; i < img_channels; ++i)
        {
            input_bgr_channels.emplace_back(
                img_size.height, img_size.width, img_type1, &(input_data[pos]));
            pos += input_bgr_channels.back().total() *
                input_bgr_channels.back().elemSize();
        }

        cv::split(sample, input_bgr_channels);

        if (pos != img_byte_size)
        {
            std::cerr << "unexpected total size of channels " << pos << ", expecting "
                << img_byte_size << std::endl;
            exit(1);
        }

        return input_data;
    }
};
