#include "constrained_llm.h"
#include <iostream>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model-path>" << std::endl;
        return 1;
    }

    try {
        LLMSession llm(argv[1], 512);

        std::cout << "=== Using XML-style tags as stop sequences ===" << std::endl << std::endl;

        std::cout << "Example 1: Stop at </think>" << std::endl;
        llm += "<think>Let me analyze this problem: ";

        GenerateOptions opts;
        opts.min_tokens = 5;
        opts.max_tokens = 50;
        opts.stop_sequences = {"</think>"};
        opts.var_name = "thinking";
        opts.temperature = 0.7f;

        llm.generate(opts);
        std::cout << "Thinking: '" << llm.get_variable("thinking") << "'" << std::endl;
        std::cout << "Full output: " << llm.get_output() << std::endl << std::endl;

        llm.clear();

        std::cout << "Example 2: Multiple XML stop tags" << std::endl;
        llm += "<response>";

        opts = GenerateOptions();
        opts.min_tokens = 3;
        opts.max_tokens = 30;
        opts.stop_sequences = {"</response>", "</answer>", "</output>"};
        opts.var_name = "response";

        llm.generate(opts);
        std::cout << "Response: '" << llm.get_variable("response") << "'" << std::endl << std::endl;

        llm.clear();

        std::cout << "Example 3: Structured reasoning with multiple sections" << std::endl;
        llm += "<reasoning>\n";
        llm += "<analysis>";

        opts = GenerateOptions();
        opts.min_tokens = 5;
        opts.max_tokens = 30;
        opts.stop_sequences = {"</analysis>"};
        opts.var_name = "analysis";
        llm.generate(opts);

        llm += "</analysis>\n<conclusion>";

        opts = GenerateOptions();
        opts.min_tokens = 3;
        opts.max_tokens = 20;
        opts.stop_sequences = {"</conclusion>"};
        opts.var_name = "conclusion";
        llm.generate(opts);

        llm += "</conclusion>\n</reasoning>";

        std::cout << "\nExtracted:" << std::endl;
        std::cout << "  Analysis: '" << llm.get_variable("analysis") << "'" << std::endl;
        std::cout << "  Conclusion: '" << llm.get_variable("conclusion") << "'" << std::endl;

        std::cout << "\nFull structured output:" << std::endl;
        std::cout << llm.get_output() << std::endl;

        llm.clear();

        std::cout << "\n=== Example 4: Chain-of-thought with thinking tags ===" << std::endl;
        llm += "Question: What is 15 * 24?\n\n";
        llm += "<thinking>\n";

        opts = GenerateOptions();
        opts.min_tokens = 10;
        opts.max_tokens = 50;
        opts.stop_sequences = {"\n</thinking>"};
        opts.var_name = "thought_process";
        llm.generate(opts);

        llm += "\n</thinking>\n\n";
        llm += "Answer: ";

        opts = GenerateOptions();
        opts.min_tokens = 1;
        opts.max_tokens = 10;
        opts.stop_sequences = {"\n", "."};
        opts.var_name = "answer";
        llm.generate(opts);

        std::cout << "Thought process: '" << llm.get_variable("thought_process") << "'" << std::endl;
        std::cout << "Answer: '" << llm.get_variable("answer") << "'" << std::endl;

        std::cout << "\n=== Key Points ===" << std::endl;
        std::cout << "- Any string can be a stop sequence: '</think>', '</answer>', etc." << std::endl;
        std::cout << "- Multiple stop sequences work: stops at first match" << std::endl;
        std::cout << "- Great for structured output formats (XML, JSON-like, etc.)" << std::endl;
        std::cout << "- Combine with min_tokens to ensure meaningful content" << std::endl;

    } catch (const std::exception & e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
