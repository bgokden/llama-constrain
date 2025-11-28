#include "constrained_llm.h"
#include <iostream>
#include <string>
#include "llama.h"

// Helper function to run a thinking loop
// Returns the final answer/output
std::string run_thinking_loop(
    LLMSession & llm,
    const std::vector<uint8_t> & cached_prompt,
    const std::string & question,
    int max_iterations = 5,
    int min_thinks = 2
) {
    // Reset to clean state with cached system prompt
    llm.load_context_from_memory(cached_prompt);

    // Add the question
    llm += "<input>" + question + "</input>\n\n";

    std::string answer;

    // Thinking loop
    for (int i = 0; i < max_iterations; i++) {
        // Decide next action: continue thinking or output answer
        std::string choice;

        if (i < min_thinks) {
            // Haven't reached minimum thinking steps - force <think>
            choice = "<think>";
            llm += choice;
            std::cout << "[Forced <think> - minimum " << min_thinks << " required]" << std::endl;
        } else {
            // Allow model to choose between continuing or outputting
            choice = llm.select({"<think>", "<output>"});
            std::cout << "[Model chose: " << choice << "]" << std::endl;
        }

        // Generate content based on choice
        if (choice == "<think>") {
            std::string thinking = llm.generate(300, {"</think>"}, 0.0f);
            std::cout << "Thinking " << (i+1) << ": " << thinking << std::endl;
        } else if (choice == "<output>") {
            answer = llm.generate(200, {"</output>"}, 0.0f);
            std::cout << "Output: " << answer << std::endl;
            break;
        }
    }

    // Force output if we exhausted iterations
    if (answer.empty()) {
        std::cout << "[Max iterations reached - forcing output]" << std::endl;
        llm += "<output>";
        answer = llm.generate(200, {"</output>"}, 0.0f);
        std::cout << "Output: " << answer << std::endl;
    }

    return answer;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model-path>" << std::endl;
        return 1;
    }

    // Disable llama.cpp logs with null callback
    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        (void) level;
        (void) text;
        (void) user_data;
    }, nullptr);

    try {
        std::cout << "=== Thinking Chat with Structured Output ===" << std::endl;
        std::cout << "\nCreating a chat system where the model:" << std::endl;
        std::cout << "1. Thinks through problems step-by-step in <think> tags" << std::endl;
        std::cout << "2. Provides final answer in <output> tags" << std::endl;
        std::cout << "3. Receives user input in <input> tags\n" << std::endl;

        // Create session with auto-caching enabled (larger context for system prompt)
        LLMSession llm(argv[1], 8192);
        llm.enable_auto_cache(true);

        // System prompt with multi-step thinking pattern (will be auto-cached)
        llm += R"(You are an AI that thinks step-by-step before answering.

IMPORTANT RULES:
1. Each <think> tag MUST be properly closed with </think>
2. After EVERY </think>, you must choose EITHER <think> OR <output>
3. NEVER write </think without the closing >
4. ALWAYS complete the closing tag: </think>
5. THINK MULTIPLE TIMES before giving your final answer
6. Use <think> to work through the problem step by step
7. Only use <output> when you have the COMPLETE final answer

PATTERN: <input>question</input> -> <think>step</think><think>step</think>...<output>answer</output>

After each </think>, choose next tag:
- <think> to continue thinking (USE THIS to work through the problem!)
- <output> to give final answer (ONLY use when you have worked out the complete answer!)

EXAMPLES:

<input>What is 15 * 24?</input>
<think>Break down: 15 * 20 = 300</think>
<think>Then 15 * 4 = 60</think>
<think>Sum: 300 + 60 = 360</think>
<output>360</output>

<input>If I have 5 apples and buy 3 more, then give away 2, how many left?</input>
<think>Start with 5 apples</think>
<think>Buy 3 more: 5 + 3 = 8 apples</think>
<think>Give away 2: 8 - 2 = 6 apples</think>
<output>6 apples</output>

<input>What is 123 + 456?</input>
<think>Add hundreds: 100 + 400 = 500</think>
<think>Add tens: 20 + 50 = 70</think>
<think>Add ones: 3 + 6 = 9</think>
<think>Combine: 500 + 70 + 9 = 579</think>
<output>579</output>

<input>If a train leaves at 2pm at 60mph and another at 3pm at 80mph, when does second catch up?</input>
<think>First train has 1 hour head start</think>
<think>First train travels 60 miles in that hour</think>
<think>Second train is 20mph faster</think>
<think>Time to catch up: 60 miles / 20 mph = 3 hours</think>
<think>So catches up at 3pm + 3 hours = 6pm</think>
<output>The second train catches up at 6pm</output>

<input>Can we conclude that some roses fade quickly if all roses are flowers and some flowers fade quickly?</input>
<think>All roses are flowers (roses ⊆ flowers)</think>
<think>Some flowers fade quickly (not all)</think>
<think>Those quick-fading flowers might not be roses</think>
<think>We can't conclude roses fade quickly without more info</think>
<output>No, we cannot conclude that. The quick-fading flowers might not be roses.</output>

<input>What color is the sky?</input>
<think>Typically asking about daytime</think>
<output>Blue</output>

<input>What is 50 + 25 + 10?</input>
<think>First add 50 + 25 = 75</think>
<think>Then add 75 + 10 = 85</think>
<output>85</output>

<input>Is 17 a prime number?</input>
<think>Check if 17 is divisible by 2: No</think>
<think>Check if 17 is divisible by 3: No</think>
<think>Check if 17 is divisible by 4: No</think>
<think>Only need to check up to sqrt(17) which is about 4.1</think>
<output>Yes, 17 is a prime number</output>

<input>If it takes 5 workers 3 hours to build a wall, how long would it take 15 workers?</input>
<think>Total work = 5 workers × 3 hours = 15 worker-hours</think>
<think>With 15 workers: 15 worker-hours ÷ 15 workers = 1 hour</think>
<output>1 hour</output>

<input>What is the capital of France?</input>
<think>This is a basic geography question</think>
<output>Paris</output>

<input>If all cats are mammals and some mammals are pets, can we conclude all cats are pets?</input>
<think>All cats are mammals (true)</think>
<think>Some mammals are pets (not all)</think>
<think>Some mammals are wild animals, not pets</think>
<think>Cats could be in either category based on this info</think>
<output>No, we cannot conclude that all cats are pets from this information alone</output>

REMEMBER: Always close tags properly with </think> and </output>
Every tag must be complete: </think> with the > at the end!

Follow this pattern exactly. Think step by step, then output your final answer.

)";

        std::cout << "System prompt loaded and cached!" << std::endl;
        std::cout << "Cache size: " << (llm.get_cached_prompt().size() / 1024.0 / 1024.0) << " MB\n" << std::endl;

        // Save the cached prompt to reuse for each example
        std::vector<uint8_t> cached_prompt = llm.get_cached_prompt();

        // Example 1: Math problem with loop
        std::cout << "=== Example 1: Math Problem (with thinking loop) ===" << std::endl;
        std::cout << "User: What is 123 + 456?" << std::endl;
        std::string answer1 = run_thinking_loop(llm, cached_prompt, "What is 123 + 456?");

        // Example 2: Reasoning question with loop
        std::cout << "\n=== Example 2: Reasoning Question (with thinking loop) ===" << std::endl;
        std::cout << "User: If a train leaves at 2pm going 60mph and another at 3pm going 80mph, when does the second catch up?" << std::endl;
        std::string answer2 = run_thinking_loop(llm, cached_prompt, "If a train leaves at 2pm going 60mph and another at 3pm going 80mph from the same station in the same direction, when does the second train catch up?");

        // Example 3: Logic puzzle with loop
        std::cout << "\n=== Example 3: Logic Puzzle (with thinking loop) ===" << std::endl;
        std::cout << "User: If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?" << std::endl;
        std::string answer3 = run_thinking_loop(llm, cached_prompt, "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?");

        // Example 4: Word problem with loop
        std::cout << "\n=== Example 4: Word Problem (with thinking loop) ===" << std::endl;
        std::cout << "User: Sarah has twice as many apples as oranges. She has 3 oranges. If she gives away half her apples, how many apples does she have left?" << std::endl;
        std::string answer4 = run_thinking_loop(llm, cached_prompt, "Sarah has twice as many apples as oranges. She has 3 oranges. If she gives away half her apples, how many apples does she have left?");

        // Example 5: Simple question (quick answer)
        std::cout << "\n=== Example 5: Simple Question (quick) ===" << std::endl;
        std::cout << "User: What color is the sky?" << std::endl;
        std::string answer5 = run_thinking_loop(llm, cached_prompt, "What color is the sky?");

    } catch (const std::exception & e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
