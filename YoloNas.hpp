#include "Yolo.hpp"
class YoloNas : public Yolo
{
public:
    YoloNas(const int input_width, const int input_height) :
        Yolo(input_width, input_height)
    {
        // Additional constructor code for YoloNas if needed
    }

    // Override the preprocess function
    std::vector<uint8_t> preprocess(
        const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
        size_t img_channels, const cv::Size& img_size) 
    {
        std::vector<uint8_t> input_data;
        cv::Mat sample;
        cv::cvtColor(img, sample,  cv::COLOR_BGR2RGB);
        sample.convertTo(
            sample, (img_channels == 3) ? img_type3 : img_type1);
        sample.convertTo(sample, CV_32FC3, 1.f / 255.f);


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
    std::vector<Detection> postprocess(const cv::Size& frame_size, std::vector<std::vector<float>>& infer_result, 
    const std::vector<std::vector<int64_t>>& infer_shapes)
    {
    
        std::vector<Detection> detections;
        std::vector<cv::Rect> boxes;
        std::vector<float> confs;
        std::vector<int> classIds;
        const auto confThreshold = 0.5f;
        const auto iouThreshold = 0.4f;

        // const int numClasses =  infer_shape[1] - 4;
        // for (auto it = infer_results.begin(); it != infer_results.end(); it += (numClasses + 5))
        // {
        //     float clsConf = it[4];
        //     if (clsConf > confThreshold)
        //     {
        //         auto[objConf, classId] = Yolo::getBestClassInfo(it, numClasses);
        //         boxes.emplace_back(Yolo::get_rect(frame_size, std::vector<float>(it, it + 4)));
        //         float confidence = clsConf * objConf;
        //         confs.emplace_back(confidence);
        //         classIds.emplace_back(classId);              
        //     }
        // }

        // std::vector<int> indices;
        // cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);

        // for (int idx : indices)
        // {
        //     Detection d;
        //     d.bbox = cv::Rect(boxes[idx]);
        //     d.class_confidence = confs[idx];
        //     d.class_id = classIds[idx];
        //     detections.emplace_back(d);
        // }        
        return detections;
    }

private:
    // Add additional member variables specific to YoloNas
};
