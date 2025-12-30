#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <algorithm> 
#include <random>
#include <vector>
#include <string>

// Common types and lightweight includes that don't require heavy dependencies
// Heavy dependencies like grpc_client.h should only be included where needed