#pragma once
// Forwarding header — canonical definition is in tritonic/infra/logger.hpp
#include "tritonic/infra/logger.hpp"

// Backward-compatibility aliases (global namespace)
using LogLevel      = tritonic::infra::LogLevel;
using ILogger       = tritonic::infra::ILogger;
using Logger        = tritonic::infra::Logger;
using LoggerManager = tritonic::infra::LoggerManager;

