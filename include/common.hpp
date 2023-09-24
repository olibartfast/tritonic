#pragma once
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#ifdef WRITE_FRAME
#include <opencv2/videoio.hpp>
#endif
#include <fstream>
#include <filesystem>
#include <chrono>
#include  <algorithm> 

#include "grpc_client.h"
#include "http_client.h"
namespace tc = triton::client;