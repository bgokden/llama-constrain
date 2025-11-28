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
        std::cout << "=== Token Debug Test ===" << std::endl;
        std::cout << "Testing how different closing tag variations are tokenized\n" << std::endl;

        LLMSession llm(argv[1], 2048);

        // Test 1: Check what the model generates after </think
        std::cout << "Test 1: What comes after '</think' ?" << std::endl;
        llm += "Pattern: <think>content</think>\nExample: <think>The answer is 4</think";
        std::string result1 = llm.generate(5, {}, 0.0f);
        std::cout << "Generated after '</think': '" << result1 << "'" << std::endl;
        std::cout << "Hex dump: ";
        for (unsigned char c : result1) {
            printf("%02X ", c);
        }
        std::cout << "\n" << std::endl;

        // Test 2: Check variations of closing
        llm.clear();
        std::cout << "Test 2: Generating with </think> stop, but allowing extra tokens" << std::endl;
        llm += "Pattern: <think>content</think>\nExample: <think>Answer is 4";
        std::string result2 = llm.generate(10, {"</think>"}, 0.0f);
        std::cout << "Generated: '" << result2 << "'" << std::endl;
        std::string full_context = llm.get_output();
        std::cout << "Full context: '" << full_context << "'" << std::endl;

        // Check if </think> is in context
        if (full_context.find("</think>") != std::string::npos) {
            std::cout << "✓ Found </think> in context" << std::endl;
        } else {
            std::cout << "✗ </think> NOT in context" << std::endl;
            std::cout << "Last 20 chars: '";
            if (full_context.length() >= 20) {
                std::cout << full_context.substr(full_context.length() - 20);
            } else {
                std::cout << full_context;
            }
            std::cout << "'" << std::endl;
        }

        // Test 3: Character by character check at the end
        std::cout << "\nTest 3: Character analysis of context end:" << std::endl;
        if (full_context.length() >= 10) {
            std::string end = full_context.substr(full_context.length() - 10);
            std::cout << "Last 10 chars: '";
            for (char c : end) {
                if (c >= 32 && c <= 126) {
                    std::cout << c;
                } else {
                    printf("\\x%02X", (unsigned char)c);
                }
            }
            std::cout << "'" << std::endl;

            std::cout << "Hex dump: ";
            for (unsigned char c : end) {
                printf("%02X ", c);
            }
            std::cout << std::endl;
        }

        // Test 4: Try generating just ">"
        llm.clear();
        std::cout << "\nTest 4: What happens with just closing >" << std::endl;
        llm += "</think";
        std::string result4 = llm.generate(3, {}, 0.0f);
        std::cout << "Generated after '</think': '" << result4 << "'" << std::endl;
        std::cout << "First char code: " << (int)(unsigned char)result4[0] << std::endl;
        if (result4[0] == '>') {
            std::cout << "✓ First char is '>'" << std::endl;
        } else {
            std::cout << "✗ First char is NOT '>'" << std::endl;
            std::cout << "Hex: ";
            for (size_t i = 0; i < std::min(result4.length(), (size_t)5); i++) {
                printf("%02X ", (unsigned char)result4[i]);
            }
            std::cout << std::endl;
        }

    } catch (const std::exception & e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
