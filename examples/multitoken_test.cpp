#include "constrained_llm.h"
#include <iostream>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model-path>" << std::endl;
        return 1;
    }

    try {
        LLMSession llm(argv[1]);

        std::cout << "=== Test 1: Multi-token options ===" << std::endl;
        llm += "The capital of France is";
        std::string result = llm.select({" Paris", " London", " New York", " Los Angeles"});
        std::cout << "Selected: " << result << std::endl;
        std::cout << "Full output: " << llm.get_output() << std::endl << std::endl;

        llm.clear();

        std::cout << "=== Test 2: Options with different token counts ===" << std::endl;
        llm += "My favorite color is";
        result = llm.select({" red", " blue", " green yellow", " dark purple"});
        std::cout << "Selected: " << result << std::endl;
        std::cout << "Full output: " << llm.get_output() << std::endl << std::endl;

        llm.clear();

        std::cout << "=== Test 3: Longer multi-token options ===" << std::endl;
        llm += "The best programming language for beginners is";
        result = llm.select({" Python", " JavaScript", " C++ for advanced", " Java for enterprise"});
        std::cout << "Selected: " << result << std::endl;
        std::cout << "Full output: " << llm.get_output() << std::endl << std::endl;

    } catch (const std::exception & e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
