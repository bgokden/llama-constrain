#include "constrained_llm.h"
#include <iostream>
#include <cstdio>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model-path>" << std::endl;
        return 1;
    }

    const char * context_file = "prompt_context.bin";

    try {
        std::cout << "=== Example 1: Save Context with Long Prompt ===" << std::endl;

        LLMSession llm(argv[1], 2048);

        std::cout << "Loading a long prompt with few-shot examples..." << std::endl;
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

        std::cout << "Saving context after processing long prompt..." << std::endl;
        if (llm.save_context(context_file)) {
            std::cout << "Context saved to: " << context_file << std::endl;
        } else {
            std::cerr << "Failed to save context!" << std::endl;
            return 1;
        }

        std::cout << "\n=== Example 2: Load Context and Continue Generation ===" << std::endl;

        LLMSession llm2(argv[1], 2048);

        std::cout << "Loading saved context..." << std::endl;
        if (!llm2.load_context(context_file)) {
            std::cerr << "Failed to load context!" << std::endl;
            return 1;
        }
        std::cout << "Context loaded successfully!" << std::endl;

        std::cout << "\nContinuing with first input:" << std::endl;
        llm2 += "Input: Emma Wilson, age 31, lives in Berlin\nOutput: ";
        std::string output1 = llm2.generate(30, {"\n"}, 0.3f);
        std::cout << "Generated: " << output1 << std::endl;

        std::cout << "\nContinuing with second input:" << std::endl;
        llm2 += "\nInput: David Martinez, age 45, lives in Madrid\nOutput: ";
        std::string output2 = llm2.generate(30, {"\n"}, 0.3f);
        std::cout << "Generated: " << output2 << std::endl;

        std::cout << "\n=== Example 3: Multiple Load/Save Cycles ===" << std::endl;

        llm2 += "\n\nThis context now includes 2 generations.";
        if (llm2.save_context("context_with_generations.bin")) {
            std::cout << "Saved context with generations" << std::endl;
        }

        LLMSession llm3(argv[1], 2048);
        if (llm3.load_context("context_with_generations.bin")) {
            std::cout << "Loaded context with previous generations" << std::endl;
            std::cout << "\nFull conversation so far:\n" << llm3.get_output() << std::endl;
        }

        std::cout << "\n=== Example 4: Pattern Generation from Loaded Context ===" << std::endl;

        LLMSession llm4(argv[1], 2048);
        if (llm4.load_context(context_file)) {
            std::cout << "Loaded original context" << std::endl;

            llm4 += "Input: Customer ID: ";

            GenerateOptions opts;
            opts.max_tokens = 2;
            opts.pattern = PATTERN_NUMERIC;
            opts.temperature = 0.3f;

            std::string customer_id = llm4.generate(opts);
            std::cout << "Generated customer ID (numeric): " << customer_id << std::endl;

            llm4 += ", Name: ";
            opts = GenerateOptions();
            opts.max_tokens = 2;
            opts.pattern = PATTERN_CAPITALIZED;
            opts.temperature = 0.5f;

            std::string name = llm4.generate(opts);
            std::cout << "Generated name (capitalized): " << name << std::endl;
        }

        std::remove(context_file);
        std::remove("context_with_generations.bin");
        std::cout << "\nCleaned up temporary files" << std::endl;

    } catch (const std::exception & e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
