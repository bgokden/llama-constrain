#include "constrained_llm.h"
#include <iostream>
#include <string>
#include "llama.h"

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model-path>" << std::endl;
        return 1;
    }

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        (void) level;
        (void) text;
        (void) user_data;
    }, nullptr);

    try {
        std::cout << "=== Testing for > prefix issue ===" << std::endl;

        LLMSession llm(argv[1], 2048);

        // Test 1: Select then generate immediately
        std::cout << "\n--- Test 1: select(<think>) then generate() ---" << std::endl;
        llm += "Test:\n";
        std::string tag = llm.select({"<think>"});
        std::cout << "Selected: [" << tag << "]" << std::endl;
        std::cout << "Context: [" << llm.get_output() << "]" << std::endl;

        std::string content = llm.generate(20, {"</think>"}, 0.0f);
        std::cout << "Generated: [" << content << "]" << std::endl;
        if (!content.empty() && content[0] == '>') {
            std::cout << "❌ FAIL: Found > prefix (char code: " << (int)content[0] << ")" << std::endl;
        } else {
            std::cout << "✓ PASS: No > prefix" << std::endl;
        }

        // Test 2: Select with immediate context addition
        std::cout << "\n--- Test 2: select(<addmemory>) then llm += '<key>' ---" << std::endl;
        llm.clear();
        llm += "Test:\n";
        tag = llm.select({"<addmemory>"});
        std::cout << "Selected: [" << tag << "]" << std::endl;
        llm += "<key>";  // Immediate context
        std::string key = llm.generate(20, {"</key>"}, 0.0f);
        std::cout << "Generated: [" << key << "]" << std::endl;
        if (!key.empty() && key[0] == '>') {
            std::cout << "❌ FAIL: Found > prefix" << std::endl;
        } else {
            std::cout << "✓ PASS: No > prefix" << std::endl;
        }

        // Test 3: Multiple selects in a row
        std::cout << "\n--- Test 3: select(<think>) -> select(<response>) ---" << std::endl;
        llm.clear();
        llm += "Test:\n";
        tag = llm.select({"<think>"});
        std::cout << "First select: [" << tag << "]" << std::endl;
        content = llm.generate(15, {"</think>"}, 0.0f);
        std::cout << "First content: [" << content << "]" << std::endl;

        tag = llm.select({"<response>"});
        std::cout << "Second select: [" << tag << "]" << std::endl;
        content = llm.generate(15, {"</response>"}, 0.0f);
        std::cout << "Second content: [" << content << "]" << std::endl;
        if (!content.empty() && content[0] == '>') {
            std::cout << "❌ FAIL: Found > prefix on second" << std::endl;
        } else {
            std::cout << "✓ PASS: No > prefix on second" << std::endl;
        }

        std::cout << "\n=== All tests complete ===" << std::endl;

    } catch (const std::exception & e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
