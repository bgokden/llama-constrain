#include "constrained_llm.h"
#include <iostream>
#include <chrono>

using namespace std::chrono;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model-path>" << std::endl;
        return 1;
    }

    std::string long_prompt = R"(You are a helpful assistant that extracts structured data.

Example 1:
Input: John Smith, age 42, lives in Paris
Output: Name=John Smith, Age=42, City=Paris

Example 2:
Input: Sarah Johnson, age 35, lives in London
Output: Name=Sarah Johnson, Age=35, City=London

Example 3:
Input: Michael Chen, age 28, lives in Tokyo
Output: Name=Michael Chen, Age=28, City=Tokyo

Now extract data from the following:
)";

    try {
        std::cout << "=== Memory-Based Context Save/Load Example ===" << std::endl;

        std::cout << "\n1. Creating session and processing prompt..." << std::endl;
        auto start1 = high_resolution_clock::now();

        LLMSession llm1(argv[1], 2048);
        llm1 += long_prompt;

        auto end1 = high_resolution_clock::now();
        auto duration1 = duration_cast<milliseconds>(end1 - start1);
        std::cout << "Time to process prompt: " << duration1.count() << " ms" << std::endl;

        std::cout << "\n2. Saving context to memory..." << std::endl;
        auto start_save = high_resolution_clock::now();

        std::vector<uint8_t> context_data = llm1.save_context_to_memory();

        auto end_save = high_resolution_clock::now();
        auto duration_save = duration_cast<milliseconds>(end_save - start_save);
        std::cout << "Saved " << context_data.size() << " bytes to memory in "
                  << duration_save.count() << " ms" << std::endl;

        std::cout << "\n3. Loading context from memory into new session..." << std::endl;
        auto start_load = high_resolution_clock::now();

        LLMSession llm2(argv[1], 2048);
        if (!llm2.load_context_from_memory(context_data)) {
            std::cerr << "Failed to load context from memory!" << std::endl;
            return 1;
        }

        auto end_load = high_resolution_clock::now();
        auto duration_load = duration_cast<milliseconds>(end_load - start_load);
        std::cout << "Loaded context from memory in " << duration_load.count() << " ms" << std::endl;

        std::cout << "\n4. Verifying both contexts work identically..." << std::endl;

        llm1 += "Input: Alice Brown, age 30, lives in Boston\nOutput: ";
        std::string output1 = llm1.generate(20, {"\n"}, 0.3f);
        std::cout << "Original session output: " << output1 << std::endl;

        llm2 += "Input: Alice Brown, age 30, lives in Boston\nOutput: ";
        std::string output2 = llm2.generate(20, {"\n"}, 0.3f);
        std::cout << "Loaded session output:   " << output2 << std::endl;

        std::cout << "\n5. Use case: Caching multiple prompt states in memory" << std::endl;
        std::cout << "You can store multiple context states without touching disk:" << std::endl;

        std::vector<uint8_t> state1 = llm1.save_context_to_memory();
        std::vector<uint8_t> state2 = llm2.save_context_to_memory();

        std::cout << "State 1 size: " << state1.size() << " bytes" << std::endl;
        std::cout << "State 2 size: " << state2.size() << " bytes" << std::endl;

        std::cout << "\nYou can now restore any state instantly without file I/O!" << std::endl;

        std::cout << "\n=== Speed Comparison ===" << std::endl;
        std::cout << "Process prompt directly: " << duration1.count() << " ms" << std::endl;
        std::cout << "Save to memory:          " << duration_save.count() << " ms" << std::endl;
        std::cout << "Load from memory:        " << duration_load.count() << " ms" << std::endl;

        float speedup = static_cast<float>(duration1.count()) / duration_load.count();
        std::cout << "Speedup:                 " << speedup << "x faster" << std::endl;

        std::cout << "\n=== Benefits of Memory-Based Save/Load ===" << std::endl;
        std::cout << "- No disk I/O overhead" << std::endl;
        std::cout << "- Can store multiple states in RAM" << std::endl;
        std::cout << "- Ideal for branching conversations" << std::endl;
        std::cout << "- Useful for A/B testing different continuations" << std::endl;
        std::cout << "- Perfect for implementing undo/redo" << std::endl;

    } catch (const std::exception & e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
