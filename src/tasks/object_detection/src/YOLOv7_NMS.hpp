#include "TaskInterface.hpp"
class YOLOv7_NMS : public TaskInterface
{
public:
    YOLOv7_NMS(int input_width, int input_height)
        : TaskInterface(input_width, input_height) {
    }

    // Override the preprocess function
    std::vector<uint8_t> preprocess(
        const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
        size_t img_channels, const cv::Size& img_size) override
    {
        std::vector<uint8_t> input_data;
        cv::Mat sample;
        cv::cvtColor(img, sample,  cv::COLOR_BGR2RGB);
        sample.convertTo(
            sample, (img_channels == 3) ? img_type3 : img_type1);
        sample.convertTo(sample, CV_32FC3, 1.f / 255.f);
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

    // Override the postprocess function
    std::vector<Result> postprocess(const cv::Size& frame_size, std::vector<std::vector<float>>& infer_results, 
    const std::vector<std::vector<int64_t>>& infer_shapes) override
    {
    
        std::vector<Result> detections;

        const auto confThreshold = 0.5f;
        const auto iouThreshold = 0.4f;
        float* output0 = infer_results[0].data();
        const auto results = infer_shapes[0][2];
        const auto data_buffer = infer_shapes[0][1];


        for (int i = 0; i < results; i++)
        {
            float r_w = (frame_size.width * 1.0) / 640;
            float r_h = (frame_size.height * 1.0) / 640 ;
            float batch_id = output0[0];
            float x0 = output0[1];
            float y0 = output0[2];
            float x1 = output0[3];
            float y1 = output0[4];
            float cls_id = output0[5];
            float score = output0[6];    
            if(score < 0.5f)
                continue;
            cv::Rect box(static_cast<int>(x0) * r_w, static_cast<int>(y0) * r_h, static_cast<int>(x1 - x0) * r_w, static_cast<int>(y1 - y0) * r_h);
            box = box & cv::Rect(0, 0, frame_size.width, frame_size.height); // Clip box to image boundaries                    
            Detection det;
            det.class_id = output0[5];
            det.class_confidence = output0[6];
            det.bbox = box;
            output0 += data_buffer;
            detections.emplace_back(Result(det));
        }
        return detections;
    } 

private:
    // Add additional member variables specific to YoloNas
};
