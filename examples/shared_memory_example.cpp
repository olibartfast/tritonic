#include "../src/triton/Triton.hpp"
#include <iostream>
#include <vector>

void demonstrateSystemSharedMemory() {
    std::cout << "\n=== System (CPU) Shared Memory Example ===" << std::endl;
    
    try {
        // Create Triton client with system shared memory (default)
        std::string server_url = "localhost";
        std::string model_name = "resnet50";
        
        Triton triton_client(server_url, ProtocolType::HTTP, model_name, "1", true, 
                           SharedMemoryType::SYSTEM_SHARED_MEMORY);
        
        // Get model information first
        auto model_info = triton_client.getModelInfo(model_name, server_url);
        
        // Register shared memory regions for inputs and outputs
        std::cout << "Registering system input shared memory..." << std::endl;
        triton_client.registerInputSharedMemory();
        
        std::cout << "Registering system output shared memory..." << std::endl;
        triton_client.registerOutputSharedMemory();
        
        // Prepare input data (example for image classification)
        std::vector<std::vector<uint8_t>> input_data;
        
        // For demonstration, create dummy input data
        for (size_t i = 0; i < model_info.input_names.size(); ++i) {
            const auto& shape = model_info.input_shapes[i];
            const auto& datatype = model_info.input_datatypes[i];
            
            size_t total_elements = 1;
            for (int64_t dim : shape) {
                total_elements *= dim;
            }
            
            size_t bytes_per_element = (datatype == "FP32") ? 4 : 1;
            size_t total_bytes = total_elements * bytes_per_element;
            
            std::vector<uint8_t> dummy_data(total_bytes, 0);
            // Fill with some dummy data
            for (size_t j = 0; j < total_bytes; ++j) {
                dummy_data[j] = static_cast<uint8_t>(j % 256);
            }
            
            input_data.push_back(std::move(dummy_data));
        }
        
        // Perform inference using system shared memory
        std::cout << "Performing inference with system shared memory..." << std::endl;
        auto [results, output_shapes] = triton_client.inferWithSharedMemory(input_data);
        
        // Process results
        std::cout << "System shared memory inference completed successfully!" << std::endl;
        std::cout << "Number of outputs: " << results.size() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "System shared memory error: " << e.what() << std::endl;
    }
}

void demonstrateCudaSharedMemory() {
    std::cout << "\n=== CUDA (GPU) Shared Memory Example ===" << std::endl;
    
    try {
        // Create Triton client with CUDA shared memory
        std::string server_url = "localhost";
        std::string model_name = "resnet50";
        int cuda_device = 0;  // Use GPU 0
        
        Triton triton_client(server_url, ProtocolType::HTTP, model_name, "1", true, 
                           SharedMemoryType::CUDA_SHARED_MEMORY, cuda_device);
        
        // Get model information first
        auto model_info = triton_client.getModelInfo(model_name, server_url);
        
        // Register CUDA shared memory regions for inputs and outputs
        std::cout << "Registering CUDA input shared memory..." << std::endl;
        triton_client.registerInputSharedMemory();
        
        std::cout << "Registering CUDA output shared memory..." << std::endl;
        triton_client.registerOutputSharedMemory();
        
        // Prepare input data
        std::vector<std::vector<uint8_t>> input_data;
        
        for (size_t i = 0; i < model_info.input_names.size(); ++i) {
            const auto& shape = model_info.input_shapes[i];
            const auto& datatype = model_info.input_datatypes[i];
            
            size_t total_elements = 1;
            for (int64_t dim : shape) {
                total_elements *= dim;
            }
            
            size_t bytes_per_element = (datatype == "FP32") ? 4 : 1;
            size_t total_bytes = total_elements * bytes_per_element;
            
            std::vector<uint8_t> dummy_data(total_bytes, 0);
            // Fill with some dummy data
            for (size_t j = 0; j < total_bytes; ++j) {
                dummy_data[j] = static_cast<uint8_t>(j % 256);
            }
            
            input_data.push_back(std::move(dummy_data));
        }
        
        // Perform inference using CUDA shared memory
        std::cout << "Performing inference with CUDA shared memory..." << std::endl;
        auto [results, output_shapes] = triton_client.inferWithSharedMemory(input_data);
        
        // Process results
        std::cout << "CUDA shared memory inference completed successfully!" << std::endl;
        std::cout << "Number of outputs: " << results.size() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "CUDA shared memory error: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "Triton Shared Memory Examples" << std::endl;
    
    // Demonstrate system (CPU) shared memory
    demonstrateSystemSharedMemory();
    
    // Demonstrate CUDA (GPU) shared memory
    demonstrateCudaSharedMemory();
    
    return 0;
}
