#include "constrained_llm.h"
#include <iostream>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model-path>" << std::endl;
        return 1;
    }

    try {
        LLMSession llm(argv[1]);

        std::cout << "=== Example 1: Simple Select ===" << std::endl;
        llm += "The capital of France is";
        std::string city = llm.select({" Paris", " London", " Berlin", " Madrid"});
        std::cout << "Output: " << llm.get_output() << std::endl << std::endl;

        llm.clear();

        std::cout << "=== Example 2: Generate with Stop ===" << std::endl;
        llm += "Q: What is 2+2?\nA:";
        std::string answer = llm.generate(30, {"\nQ:", "\n\n"});
        std::cout << "Output: " << llm.get_output() << std::endl << std::endl;

        llm.clear();

        std::cout << "=== Example 3: Multi-turn Conversation ===" << std::endl;
        llm += "Q: Name a programming language\nA:";
        std::string lang = llm.select({" Python", " JavaScript", " C++", " Rust"});
        llm += "\n\nQ: Is " + lang + " good for beginners?\nA:";
        std::string opinion = llm.generate(20, {"\n"});
        std::cout << "Output: " << llm.get_output() << std::endl << std::endl;

        llm.clear();

        std::cout << "=== Example 4: Chain Multiple Operations ===" << std::endl;
        llm += "Story starter: Once upon a time";
        llm.generate(30);
        llm += "\n\nWhat genre is this?";
        std::string genre = llm.select({" Fantasy", " Sci-Fi", " Mystery", " Romance"});
        std::cout << "Output: " << llm.get_output() << std::endl << std::endl;

    } catch (const std::exception & e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
