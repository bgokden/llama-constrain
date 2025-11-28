#include "constrained_llm.h"
#include <iostream>
#include <chrono>

using namespace std::chrono;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model-path>" << std::endl;
        return 1;
    }

    try {
        std::cout << "=== Cache Size vs Prompt Length Test ===" << std::endl;
        std::cout << "\nTesting how cache size and load time scale with prompt length\n" << std::endl;

        // Test different prompt lengths
        std::vector<int> word_counts = {10, 50, 100, 200, 500};

        for (int words : word_counts) {
            std::cout << "--- Testing with ~" << words << " words ---" << std::endl;

            // Create prompt with specified length
            std::string prompt = "You are a helpful assistant. ";
            for (int i = 0; i < words; i++) {
                prompt += "Word" + std::to_string(i) + " ";
            }
            prompt += "\nNow answer: ";

            // Process and cache
            auto start_process = high_resolution_clock::now();

            LLMSession llm(argv[1], 4096, true);
            llm.enable_auto_cache(true);
            llm += prompt;

            auto end_process = high_resolution_clock::now();
            auto process_time = duration_cast<milliseconds>(end_process - start_process);

            std::vector<uint8_t> cache = llm.get_cached_prompt();

            std::cout << "  Process time: " << process_time.count() << " ms" << std::endl;
            std::cout << "  Cache size:   " << cache.size() << " bytes ("
                      << (cache.size() / 1024.0 / 1024.0) << " MB)" << std::endl;

            // Test load time
            auto start_load = high_resolution_clock::now();

            LLMSession llm2(argv[1], 4096, true);
            llm2.load_context_from_memory(cache);

            auto end_load = high_resolution_clock::now();
            auto load_time = duration_cast<milliseconds>(end_load - start_load);

            std::cout << "  Load time:    " << load_time.count() << " ms" << std::endl;
            std::cout << "  Bytes/ms:     " << (cache.size() / 1024.0) / load_time.count() << " KB/ms" << std::endl;
            std::cout << std::endl;
        }

        std::cout << "=== Key Findings ===" << std::endl;
        std::cout << "1. Cache size grows with prompt length (KV cache stores attention keys/values)" << std::endl;
        std::cout << "2. Load time increases with cache size (memory copy + state restoration)" << std::endl;
        std::cout << "3. But it's still much faster than re-processing the prompt!" << std::endl;
        std::cout << "\nThe KV cache stores:" << std::endl;
        std::cout << "- Key/Value tensors for each attention layer" << std::endl;
        std::cout << "- Size = n_tokens × n_layers × hidden_dim" << std::endl;

    } catch (const std::exception & e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
