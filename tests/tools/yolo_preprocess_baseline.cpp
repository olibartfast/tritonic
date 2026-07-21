#include "neuriplo/tasks/core/vision/opencv_adapter.hpp"
#include "neuriplo/tasks/object_detection/detection_preprocessor.hpp"

#include <opencv2/imgcodecs.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: yolo_preprocess_baseline OUTPUT_DIR IMAGE [IMAGE ...]\n";
        return 2;
    }

    const std::filesystem::path output_dir(argv[1]);
    std::filesystem::create_directories(output_dir);
    neuriplo_tasks::YoloPreprocessor preprocessor({640, 640});

    try {
        std::cout << "#opencv" << '\t' << CV_VERSION << '\n';
        for (int index = 2; index < argc; ++index) {
            const std::filesystem::path source(argv[index]);
            const cv::Mat image = cv::imread(source.string());
            if (image.empty()) {
                throw std::runtime_error("unable to decode image: " + source.string());
            }

            const auto tensor =
                preprocessor.preprocess(neuriplo_tasks::vision::opencv::toImageView(image));
            const auto output = output_dir / (source.stem().string() + ".fp32");
            std::ofstream stream(output, std::ios::binary);
            if (!stream) {
                throw std::runtime_error("unable to open output: " + output.string());
            }
            stream.write(reinterpret_cast<const char*>(tensor.data()),
                         static_cast<std::streamsize>(tensor.size()));
            if (!stream) {
                throw std::runtime_error("unable to write output: " + output.string());
            }

            std::cout << source.string() << '\t' << image.cols << '\t' << image.rows << '\t'
                      << output.string() << '\t' << tensor.size() << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    return 0;
}
