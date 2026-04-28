#pragma once
// Forwarding header — canonical definition is in tritonic/core/types.hpp
#include "tritonic/core/types.hpp"

// Backward-compatibility aliases (global namespace)
using TensorElement = tritonic::core::TensorElement;
using Tensor = tritonic::core::Tensor;
using Message = tritonic::core::Message;
using ChatRequest = tritonic::core::ChatRequest;
using ChatResponse = tritonic::core::ChatResponse;
using TritonInferRequest = tritonic::core::TritonInferRequest;
using BackendRequest = tritonic::core::BackendRequest;
using BackendResponse = tritonic::core::BackendResponse;
