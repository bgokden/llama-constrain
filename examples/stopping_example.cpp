#include "constrained_llm.h"
#include <iostream>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model-path>" << std::endl;
        return 1;
    }

    try {
        LLMSession llm(argv[1], 512);

        std::cout << "=== Problem: How do we know when to stop? ===" << std::endl << std::endl;

        std::cout << "Example 1: Without stop sequences (generates max_tokens)" << std::endl;
        llm += "The capital of France is ";

        GenerateOptions opts;
        opts.max_tokens = 5;
        opts.pattern = PATTERN_CAPITALIZED;
        opts.var_name = "city1";
        opts.temperature = 0.5f;

        llm.generate(opts);
        std::cout << "Generated: '" << llm.get_variable("city1") << "'" << std::endl;
        std::cout << "Problem: Generates exactly 5 tokens, may run past the city name!\n" << std::endl;

        llm.clear();

        std::cout << "Example 2: With stop sequences (stops at boundary)" << std::endl;
        llm += "The capital of France is ";

        opts = GenerateOptions();
        opts.max_tokens = 10;
        opts.pattern = PATTERN_CAPITALIZED;
        opts.stop_sequences = {" ", ",", ".", "\n"};
        opts.var_name = "city2";
        opts.temperature = 0.5f;

        llm.generate(opts);
        std::cout << "Generated: '" << llm.get_variable("city2") << "'" << std::endl;
        std::cout << "Solution: Stops when hitting space/punctuation!\n" << std::endl;

        llm.clear();

        std::cout << "Example 3: Phone number with dash as stop" << std::endl;
        llm += "Call us: 555-";

        opts = GenerateOptions();
        opts.max_tokens = 10;
        opts.pattern = PATTERN_NUMERIC;
        opts.stop_sequences = {" ", "-", "\n"};
        opts.var_name = "phone";
        opts.temperature = 0.3f;

        llm.generate(opts);
        std::cout << "Generated: '" << llm.get_variable("phone") << "'" << std::endl;
        std::cout << "Full: " << llm.get_output() << std::endl << std::endl;

        llm.clear();

        std::cout << "Example 4: Multi-field form with proper stops" << std::endl;
        llm += "Name: ";

        opts = GenerateOptions();
        opts.max_tokens = 10;
        opts.pattern = PATTERN_CAPITALIZED;
        opts.stop_sequences = {"\n", ","};
        opts.var_name = "name";
        llm.generate(opts);

        llm += "\nAge: ";
        opts = GenerateOptions();
        opts.max_tokens = 10;
        opts.pattern = PATTERN_NUMERIC;
        opts.stop_sequences = {"\n", " "};
        opts.var_name = "age";
        llm.generate(opts);

        llm += "\nCity: ";
        opts = GenerateOptions();
        opts.max_tokens = 10;
        opts.pattern = PATTERN_CAPITALIZED;
        opts.stop_sequences = {"\n", ","};
        opts.var_name = "city";
        llm.generate(opts);

        std::cout << "\nExtracted data:" << std::endl;
        std::cout << "  Name: '" << llm.get_variable("name") << "'" << std::endl;
        std::cout << "  Age: '" << llm.get_variable("age") << "'" << std::endl;
        std::cout << "  City: '" << llm.get_variable("city") << "'" << std::endl;

        std::cout << "\nFull output:" << std::endl;
        std::cout << llm.get_output() << std::endl;

        std::cout << "\n=== Key Insight ===" << std::endl;
        std::cout << "Combine patterns + stop_sequences for best results:" << std::endl;
        std::cout << "  - Pattern: Enforces format (NUMERIC, CAPITALIZED, etc.)" << std::endl;
        std::cout << "  - Stop sequences: Defines boundaries (space, comma, newline)" << std::endl;
        std::cout << "  - max_tokens: Safety limit (prevents runaway generation)" << std::endl;

    } catch (const std::exception & e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
