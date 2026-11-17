#pragma once

#include <algorithm>
#include <csetjmp>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <jpeglib.h>

#include "neuriplo/tasks/core/vision/image.hpp"
#include "neuriplo/tasks/core/vision/stb_io.hpp"

namespace tritonic::core {

using Image = neuriplo_tasks::vision::Image;
using PixelType = neuriplo_tasks::vision::PixelType;

namespace detail {

struct JpegErrorManager {
    jpeg_error_mgr base;
    std::jmp_buf jump;
};

inline void JpegErrorExit(j_common_ptr cinfo) {
    auto* manager = reinterpret_cast<JpegErrorManager*>(cinfo->err);
    longjmp(manager->jump, 1);
}

inline void RequireBgr8(const Image& image, const char* operation) {
    if (image.empty()) {
        throw std::runtime_error(std::string(operation) + ": image is empty");
    }
    if (image.channels() != 3 || image.pixelType() != PixelType::UInt8) {
        throw std::runtime_error(std::string(operation) + ": expected 8-bit BGR image");
    }
}

template <typename T>
inline Image ResizeBilinearTyped(const Image& source, int width, int height) {
    const int source_width = source.width();
    const int source_height = source.height();
    const int channels = source.channels();

    Image destination = Image::uninit(width, height, channels, source.pixelType());

    for (int y = 0; y < height; ++y) {
        const float source_y = (static_cast<float>(y) + 0.5F) * static_cast<float>(source_height) /
                                   static_cast<float>(height) -
                               0.5F;
        const int y0 = std::clamp(static_cast<int>(source_y), 0, source_height - 1);
        const int y1 = std::min(y0 + 1, source_height - 1);
        const float wy = std::clamp(source_y - static_cast<float>(y0), 0.0F, 1.0F);

        const T* row0 = source.template ptr<T>(y0);
        const T* row1 = source.template ptr<T>(y1);
        T* output = destination.template ptr<T>(y);

        for (int x = 0; x < width; ++x) {
            const float source_x = (static_cast<float>(x) + 0.5F) *
                                       static_cast<float>(source_width) /
                                       static_cast<float>(width) -
                                   0.5F;
            const int x0 = std::clamp(static_cast<int>(source_x), 0, source_width - 1);
            const int x1 = std::min(x0 + 1, source_width - 1);
            const float wx = std::clamp(source_x - static_cast<float>(x0), 0.0F, 1.0F);

            for (int c = 0; c < channels; ++c) {
                const float top = static_cast<float>(row0[x0 * channels + c]) * (1.0F - wx) +
                                  static_cast<float>(row0[x1 * channels + c]) * wx;
                const float bottom = static_cast<float>(row1[x0 * channels + c]) * (1.0F - wx) +
                                     static_cast<float>(row1[x1 * channels + c]) * wx;
                const float value = top * (1.0F - wy) + bottom * wy;
                output[x * channels + c] = static_cast<T>(value + 0.5F);
            }
        }
    }

    return destination;
}

}  // namespace detail

/**
 * @brief Load an image from disk as an 8-bit, three-channel BGR image.
 *
 * neuriplo-tasks' stb loader returns RGB; tritonic internal images are BGR so
 * they can be fed to preprocessors without another conversion.
 */
[[nodiscard]] inline Image LoadBgrImage(const std::string& path) {
    Image image = neuriplo_tasks::vision::loadImage(path, 3);
    if (image.channels() == 3) {
        for (int y = 0; y < image.height(); ++y) {
            std::uint8_t* row = image.ptr<std::uint8_t>(y);
            for (int x = 0; x < image.width(); ++x) {
                std::swap(row[x * 3], row[x * 3 + 2]);
            }
        }
    }
    return image;
}

/// Save an 8-bit BGR image to disk, swapping to RGB for stb's writer.
inline bool SaveBgrImage(const std::string& path, const Image& image) {
    detail::RequireBgr8(image, "SaveBgrImage");
    Image rgb = image;
    for (int y = 0; y < rgb.height(); ++y) {
        std::uint8_t* row = rgb.ptr<std::uint8_t>(y);
        for (int x = 0; x < rgb.width(); ++x) {
            std::swap(row[x * 3], row[x * 3 + 2]);
        }
    }
    return neuriplo_tasks::vision::saveImage(path, rgb);
}

/// Encode an 8-bit BGR image as a JPEG byte stream.
[[nodiscard]] inline std::vector<std::uint8_t> EncodeJpeg(const Image& image, int quality = 90) {
    detail::RequireBgr8(image, "EncodeJpeg");

    Image rgb = image;
    for (int y = 0; y < rgb.height(); ++y) {
        std::uint8_t* row = rgb.ptr<std::uint8_t>(y);
        for (int x = 0; x < rgb.width(); ++x) {
            std::swap(row[x * 3], row[x * 3 + 2]);
        }
    }

    jpeg_compress_struct compressor{};
    detail::JpegErrorManager error_manager{};
    compressor.err = jpeg_std_error(&error_manager.base);
    error_manager.base.error_exit = detail::JpegErrorExit;

    if (setjmp(error_manager.jump) != 0) {
        jpeg_destroy_compress(&compressor);
        throw std::runtime_error("EncodeJpeg: libjpeg failed to encode image");
    }

    unsigned char* buffer = nullptr;
    unsigned long buffer_size = 0;

    jpeg_create_compress(&compressor);
    jpeg_mem_dest(&compressor, &buffer, &buffer_size);

    compressor.image_width = static_cast<JDIMENSION>(rgb.width());
    compressor.image_height = static_cast<JDIMENSION>(rgb.height());
    compressor.input_components = 3;
    compressor.in_color_space = JCS_RGB;
    jpeg_set_defaults(&compressor);
    jpeg_set_quality(&compressor, quality, TRUE);
    jpeg_start_compress(&compressor, TRUE);

    while (compressor.next_scanline < compressor.image_height) {
        std::uint8_t* row = rgb.ptr<std::uint8_t>(static_cast<int>(compressor.next_scanline));
        JSAMPROW rows[] = {row};
        jpeg_write_scanlines(&compressor, rows, 1);
    }

    jpeg_finish_compress(&compressor);
    jpeg_destroy_compress(&compressor);

    std::vector<std::uint8_t> encoded(buffer, buffer + buffer_size);
    std::free(buffer);
    return encoded;
}

/// Bilinear resize for 8-bit or float images. Any channel count is supported.
[[nodiscard]] inline Image ResizeBilinear(const Image& source, int width, int height) {
    if (source.empty()) {
        throw std::runtime_error("ResizeBilinear: source image is empty");
    }
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("ResizeBilinear: target size must be positive");
    }
    if (source.pixelType() == PixelType::UInt8) {
        return detail::ResizeBilinearTyped<std::uint8_t>(source, width, height);
    }
    if (source.pixelType() == PixelType::Float32) {
        return detail::ResizeBilinearTyped<float>(source, width, height);
    }
    throw std::runtime_error("ResizeBilinear: unsupported pixel type");
}

/// Swap the red and blue channels in place (BGR <-> RGB).
inline void SwapRedBlue(Image& image) {
    detail::RequireBgr8(image, "SwapRedBlue");
    for (int y = 0; y < image.height(); ++y) {
        std::uint8_t* row = image.ptr<std::uint8_t>(y);
        for (int x = 0; x < image.width(); ++x) {
            std::swap(row[x * 3], row[x * 3 + 2]);
        }
    }
}

/// Copy one full row between compatible images.
inline void CopyRow(const Image& source, int source_row, Image& destination, int destination_row) {
    if (source.width() != destination.width() || source.channels() != destination.channels() ||
        source.pixelType() != destination.pixelType()) {
        throw std::runtime_error("CopyRow: images have incompatible layouts");
    }
    if (source_row < 0 || source_row >= source.height() || destination_row < 0 ||
        destination_row >= destination.height()) {
        throw std::runtime_error("CopyRow: row index out of range");
    }
    std::memcpy(destination.template ptr<std::uint8_t>(destination_row),
                source.template ptr<std::uint8_t>(source_row), source.stride());
}

/// Copy a rectangular region between compatible images.
inline void CopyRegion(const Image& source, int x, int y, int width, int height, Image& destination,
                       int destination_x, int destination_y) {
    if (source.channels() != destination.channels() ||
        source.pixelType() != destination.pixelType()) {
        throw std::runtime_error("CopyRegion: images have incompatible layouts");
    }
    if (width <= 0 || height <= 0 || x < 0 || y < 0 || destination_x < 0 || destination_y < 0 ||
        x + width > source.width() || y + height > source.height() ||
        destination_x + width > destination.width() ||
        destination_y + height > destination.height()) {
        throw std::runtime_error("CopyRegion: region out of range");
    }

    const std::size_t bytes_per_pixel = static_cast<std::size_t>(source.channels()) *
                                        neuriplo_tasks::vision::pixelTypeSize(source.pixelType());
    const std::size_t row_bytes = static_cast<std::size_t>(width) * bytes_per_pixel;

    for (int row = 0; row < height; ++row) {
        const std::uint8_t* source_row = source.template ptr<std::uint8_t>(y + row) +
                                         static_cast<std::size_t>(x) * bytes_per_pixel;
        std::uint8_t* destination_row =
            destination.template ptr<std::uint8_t>(destination_y + row) +
            static_cast<std::size_t>(destination_x) * bytes_per_pixel;
        std::memcpy(destination_row, source_row, row_bytes);
    }
}

}  // namespace tritonic::core
