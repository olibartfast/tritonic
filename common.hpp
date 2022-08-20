#pragma once
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <fstream>
#include <filesystem>

#include "grpc_client.h"
#include "http_client.h"
#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>
namespace tc = triton::client;