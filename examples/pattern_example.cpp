#include "constrained_llm.h"
#include <iostream>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model-path>" << std::endl;
        return 1;
    }

    try {
        LLMSession llm(argv[1], 512);

        std::cout << "=== Example 1: Extract Phone Number (Numeric) ===" << std::endl;
        llm += "Customer Service: Call us at 555-";

        std::cout << "Prompt: 'Customer Service: Call us at 555-'" << std::endl;
        std::cout << "Pattern: NUMERIC (digits only)" << std::endl;

        GenerateOptions opts;
        opts.max_tokens = 2;
        opts.pattern = PATTERN_NUMERIC;
        opts.var_name = "phone";
        opts.temperature = 0.3f;

        llm.generate(opts);
        std::cout << "Generated: '" << llm.get_variable("phone") << "'" << std::endl;
        std::cout << "Full output: " << llm.get_output() << std::endl << std::endl;

        llm.clear();

        std::cout << "=== Example 2: Extract City Name (Capitalized) ===" << std::endl;
        llm += "The Eiffel Tower is located in ";

        std::cout << "Prompt: 'The Eiffel Tower is located in '" << std::endl;
        std::cout << "Pattern: CAPITALIZED (starts with uppercase letter)" << std::endl;

        opts = GenerateOptions();
        opts.max_tokens = 2;
        opts.pattern = PATTERN_CAPITALIZED;
        opts.var_name = "city";
        opts.temperature = 0.5f;

        llm.generate(opts);
        std::cout << "Generated: '" << llm.get_variable("city") << "'" << std::endl;
        std::cout << "Full output: " << llm.get_output() << std::endl << std::endl;

        llm.clear();

        std::cout << "=== Example 3: Extract Username (Alpha only) ===" << std::endl;
        llm += "New user registered: john";

        std::cout << "Prompt: 'New user registered: john'" << std::endl;
        std::cout << "Pattern: ALPHA (letters only)" << std::endl;

        opts = GenerateOptions();
        opts.max_tokens = 2;
        opts.pattern = PATTERN_ALPHA;
        opts.var_name = "username";
        opts.temperature = 0.7f;

        llm.generate(opts);
        std::cout << "Generated: '" << llm.get_variable("username") << "'" << std::endl;
        std::cout << "Full output: " << llm.get_output() << std::endl << std::endl;

        llm.clear();

        std::cout << "=== Example 4: Extract Country Code (Uppercase) ===" << std::endl;
        llm += "Shipping to country code: ";

        std::cout << "Prompt: 'Shipping to country code: '" << std::endl;
        std::cout << "Pattern: UPPERCASE (uppercase letters only)" << std::endl;

        opts = GenerateOptions();
        opts.max_tokens = 1;
        opts.pattern = PATTERN_UPPERCASE;
        opts.var_name = "country_code";
        opts.temperature = 0.3f;

        llm.generate(opts);
        std::cout << "Generated: '" << llm.get_variable("country_code") << "'" << std::endl;
        std::cout << "Full output: " << llm.get_output() << std::endl << std::endl;

        llm.clear();

        std::cout << "=== Example 5: User Registration Form ===" << std::endl;
        std::cout << "Prompt includes expected format hints for the model\n" << std::endl;

        llm += "Fill out this registration form (numbers only for age and zip):\n";
        llm += "Age (e.g., 25): ";

        std::cout << "Extracting Age (NUMERIC)..." << std::endl;
        opts = GenerateOptions();
        opts.max_tokens = 1;
        opts.pattern = PATTERN_NUMERIC;
        opts.var_name = "age";
        llm.generate(opts);

        llm += "\nZip Code (e.g., 94102): ";
        std::cout << "Extracting Zip Code (NUMERIC)..." << std::endl;
        opts = GenerateOptions();
        opts.max_tokens = 2;
        opts.pattern = PATTERN_NUMERIC;
        opts.var_name = "zip";
        llm.generate(opts);

        llm += "\nFirst Name (capitalized, e.g., John): ";
        std::cout << "Extracting First Name (CAPITALIZED)..." << std::endl;
        opts = GenerateOptions();
        opts.max_tokens = 2;
        opts.pattern = PATTERN_CAPITALIZED;
        opts.var_name = "first_name";
        llm.generate(opts);

        std::cout << "\nExtracted form data:" << std::endl;
        auto vars = llm.get_variables();
        for (const auto & pair : vars) {
            std::cout << "  " << pair.first << " = '" << pair.second << "'" << std::endl;
        }
        std::cout << "\nFull conversation:" << std::endl;
        std::cout << llm.get_output() << std::endl;

    } catch (const std::exception & e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
