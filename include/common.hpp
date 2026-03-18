#pragma once
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#ifdef WRITE_FRAME
#include <opencv2/videoio.hpp>
#endif
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>

#include "grpc_client.h"
#include "http_client.h"
namespace tc = triton::client;