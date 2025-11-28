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
        std::cout << "=== Automatic Prompt Caching Example ===" << std::endl;
        std::cout << "\nWhen auto-cache is enabled, the FIRST prompt is automatically cached.\n" << std::endl;

        std::cout << "Creating LLM session and enabling auto-cache..." << std::endl;
        LLMSession llm(argv[1], 2048);
        llm.enable_auto_cache(true);

        std::cout << "\nAdding first prompt (will be automatically cached)..." << std::endl;

        auto start = high_resolution_clock::now();

        llm += R"(You are a helpful assistant that extracts structured data.

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

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);

        std::cout << "Prompt processed in " << duration.count() << " ms" << std::endl;
        std::cout << "Auto-cached: " << (llm.has_cached_prompt() ? "YES" : "NO") << std::endl;

        if (llm.has_cached_prompt()) {
            std::vector<uint8_t> cached = llm.get_cached_prompt();
            std::cout << "Cached prompt size: " << cached.size() << " bytes" << std::endl;
            std::cout << "\nNOTE: Cache was created automatically after first +=\n" << std::endl;
        }

        std::cout << "\n=== First Query ===" << std::endl;
        llm += "Input: Alice Brown, age 30, lives in Boston\nOutput: ";
        std::string output1 = llm.generate(20, {"\n"}, 0.3f);
        std::cout << "Result: " << output1 << std::endl;

        std::cout << "\n=== Reusing with Cached Prompt ===" << std::endl;
        std::cout << "Creating new session and loading from cache..." << std::endl;

        auto start_cached = high_resolution_clock::now();

        LLMSession llm2(argv[1], 2048);
        std::vector<uint8_t> cached_data = llm.get_cached_prompt();
        llm2.load_context_from_memory(cached_data);

        auto end_cached = high_resolution_clock::now();
        auto duration_cached = duration_cast<milliseconds>(end_cached - start_cached);

        std::cout << "Loaded cached prompt in " << duration_cached.count() << " ms" << std::endl;

        llm2 += "Input: Bob Wilson, age 45, lives in Seattle\nOutput: ";
        std::string output2 = llm2.generate(20, {"\n"}, 0.3f);
        std::cout << "Result: " << output2 << std::endl;

        std::cout << "\n=== Third Query with Same Cache ===" << std::endl;
        LLMSession llm3(argv[1], 2048);
        llm3.load_context_from_memory(cached_data);
        llm3 += "Input: Carol Davis, age 33, lives in Austin\nOutput: ";
        std::string output3 = llm3.generate(20, {"\n"}, 0.3f);
        std::cout << "Result: " << output3 << std::endl;

        std::cout << "\n=== Complete Workflow ===" << std::endl;
        std::cout << "Step 1: llm.enable_auto_cache(true)" << std::endl;
        std::cout << "Step 2: llm += \"your prompt\"  // Automatically cached!" << std::endl;
        std::cout << "Step 3: cache = llm.get_cached_prompt()" << std::endl;
        std::cout << "Step 4: For each request:" << std::endl;
        std::cout << "        - Create new session" << std::endl;
        std::cout << "        - session.load_context_from_memory(cache)" << std::endl;
        std::cout << "        - session += user_input" << std::endl;
        std::cout << "        - session.generate(...)" << std::endl;
        std::cout << "\nBenefit: Process prompt once, reuse many times!" << std::endl;

        std::cout << "\n=== Production Example ===" << std::endl;
        std::cout << R"(
// Setup once at startup:
LLMSession template_session(model_path);
template_session.enable_auto_cache(true);
template_session += "System prompt with examples...";
std::vector<uint8_t> prompt_cache = template_session.get_cached_prompt();

// For each user request (in handler/thread):
LLMSession session(model_path);
session.load_context_from_memory(prompt_cache);  // Fast!
session += user_input;
std::string response = session.generate(100);
)" << std::endl;

    } catch (const std::exception & e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
