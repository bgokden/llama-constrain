#include "constrained_llm.h"
#include <iostream>
#include <string>
#include "llama.h"

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model-path>" << std::endl;
        return 1;
    }

    // Disable llama.cpp logs
    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        (void) level;
        (void) text;
        (void) user_data;
    }, nullptr);

    try {
        std::cout << "=== Stop Sequence Test ===" << std::endl;
        std::cout << "Testing if stop sequences are properly added to context\n" << std::endl;

        LLMSession llm(argv[1], 2048);

        // Test 1: Simple stop sequence
        std::cout << "Test 1: Generate with </think> stop sequence" << std::endl;
        llm += "Pattern: <think>content</think>\nExample: <think>The answer is 4";
        std::string result1 = llm.generate(50, {"</think>"}, 0.7f);
        std::cout << "Generated: " << result1 << std::endl;
        std::cout << "Length: " << result1.length() << " chars" << std::endl;

        std::string output1 = llm.get_output();
        std::cout << "Full context: " << output1 << std::endl;

        if (output1.find("</think>") != std::string::npos) {
            std::cout << "✓ Stop sequence </think> found in context!" << std::endl;
        } else {
            std::cout << "✗ Stop sequence </think> NOT found in context!" << std::endl;
        }

        // Test 2: Multi-word stop sequence
        std::cout << "\nTest 2: Generate with multi-word stop sequence" << std::endl;
        llm.clear();
        llm += "The answer is: ";
        std::string result2 = llm.generate(50, {"END OF ANSWER"}, 0.7f);
        std::cout << "Generated: " << result2 << std::endl;

        std::string output2 = llm.get_output();
        std::cout << "Full context: " << output2 << std::endl;

        if (output2.find("END OF ANSWER") != std::string::npos) {
            std::cout << "✓ Stop sequence 'END OF ANSWER' found in context!" << std::endl;
        } else {
            std::cout << "✗ Stop sequence 'END OF ANSWER' NOT found in context!" << std::endl;
        }

        // Test 3: XML-like tag
        std::cout << "\nTest 3: Generate with </output> stop sequence" << std::endl;
        llm.clear();
        llm += "The final result is: <output>";
        std::string result3 = llm.generate(50, {"</output>"}, 0.7f);
        std::cout << "Generated: " << result3 << std::endl;

        std::string output3 = llm.get_output();
        std::cout << "Full context: " << output3 << std::endl;

        if (output3.find("</output>") != std::string::npos) {
            std::cout << "✓ Stop sequence </output> found in context!" << std::endl;
        } else {
            std::cout << "✗ Stop sequence </output> NOT found in context!" << std::endl;
        }

        // Test 4: Check that subsequent generation works correctly
        std::cout << "\nTest 4: Continue generation after stop sequence" << std::endl;
        llm.clear();
        llm += "<think>";
        std::string think1 = llm.generate(30, {"</think>"}, 0.7f);
        std::cout << "First think: " << think1 << std::endl;

        llm += "\n<think>";
        std::string think2 = llm.generate(30, {"</think>"}, 0.7f);
        std::cout << "Second think: " << think2 << std::endl;

        std::string final_output = llm.get_output();
        std::cout << "Full context: " << final_output << std::endl;

        size_t count = 0;
        size_t pos = 0;
        while ((pos = final_output.find("</think>", pos)) != std::string::npos) {
            count++;
            pos += 8;
        }

        std::cout << "Found " << count << " </think> tags in context" << std::endl;
        if (count == 2) {
            std::cout << "✓ Both stop sequences properly added!" << std::endl;
        } else {
            std::cout << "✗ Expected 2 </think> tags, found " << count << std::endl;
        }

        // Test 5: Auto-complete partial stop sequence
        std::cout << "\nTest 5: Auto-complete partial stop sequence on max tokens" << std::endl;
        llm.clear();
        llm += "Pattern: <think>content</think>\nExample: <think>This is a very long text that will exceed the token limit and end with partial";
        // Use very small token limit to force cutoff
        std::string result5 = llm.generate(5, {"</think>"}, 0.7f);
        std::cout << "Generated (5 tokens): " << result5 << std::endl;

        std::string output5 = llm.get_output();
        std::cout << "Full context: " << output5 << std::endl;

        if (output5.find("</think>") != std::string::npos) {
            std::cout << "✓ Stop sequence auto-completed when hitting token limit!" << std::endl;
        } else {
            std::cout << "✗ Stop sequence NOT auto-completed" << std::endl;
        }

    } catch (const std::exception & e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
