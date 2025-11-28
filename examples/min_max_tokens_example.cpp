#include "constrained_llm.h"
#include <iostream>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model-path>" << std::endl;
        return 1;
    }

    try {
        LLMSession llm(argv[1], 512);

        std::cout << "=== Demonstrating min_tokens and stop_sequences ===" << std::endl << std::endl;

        std::cout << "Example 1: Extract city name with stop" << std::endl;
        llm += "The capital of France is ";

        GenerateOptions opts;
        opts.min_tokens = 1;
        opts.max_tokens = 10;
        opts.pattern = PATTERN_CAPITALIZED;
        opts.stop_sequences = {" ", ",", ".", "\n"};
        opts.var_name = "city";

        llm.generate(opts);
        std::cout << "City: '" << llm.get_variable("city") << "'" << std::endl;
        std::cout << "Full: " << llm.get_output() << std::endl << std::endl;

        llm.clear();

        std::cout << "Example 2: Extract phone number (stops at space)" << std::endl;
        llm += "Phone: ";

        opts = GenerateOptions();
        opts.min_tokens = 3;
        opts.max_tokens = 10;
        opts.pattern = PATTERN_NUMERIC;
        opts.stop_sequences = {" ", "\n", "-"};
        opts.var_name = "phone";

        llm.generate(opts);
        std::cout << "Phone: '" << llm.get_variable("phone") << "'" << std::endl << std::endl;

        llm.clear();

        std::cout << "Example 3: Form with min/max constraints" << std::endl;
        llm += "Age: ";

        opts = GenerateOptions();
        opts.min_tokens = 1;
        opts.max_tokens = 2;
        opts.pattern = PATTERN_NUMERIC;
        opts.stop_sequences = {"\n", " "};
        opts.var_name = "age";
        llm.generate(opts);

        llm += "\nZip: ";
        opts = GenerateOptions();
        opts.min_tokens = 3;
        opts.max_tokens = 5;
        opts.pattern = PATTERN_NUMERIC;
        opts.stop_sequences = {"\n", " "};
        opts.var_name = "zip";
        llm.generate(opts);

        llm += "\nName: ";
        opts = GenerateOptions();
        opts.min_tokens = 2;
        opts.max_tokens = 8;
        opts.pattern = PATTERN_CAPITALIZED;
        opts.stop_sequences = {"\n", ","};
        opts.var_name = "name";
        llm.generate(opts);

        std::cout << "\nExtracted:" << std::endl;
        std::cout << "  Age: '" << llm.get_variable("age") << "'" << std::endl;
        std::cout << "  Zip: '" << llm.get_variable("zip") << "'" << std::endl;
        std::cout << "  Name: '" << llm.get_variable("name") << "'" << std::endl;

        std::cout << "\nFull output:" << std::endl;
        std::cout << llm.get_output() << std::endl;

        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "min_tokens: Ensures minimum length (e.g., zip code needs at least 5 digits)" << std::endl;
        std::cout << "max_tokens: Safety limit (prevents runaway generation)" << std::endl;
        std::cout << "stop_sequences: Natural boundaries (space, comma, newline)" << std::endl;
        std::cout << "pattern: Format enforcement (NUMERIC, CAPITALIZED, etc.)" << std::endl;

    } catch (const std::exception & e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
