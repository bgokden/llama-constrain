#include "constrained_llm.h"
#include <iostream>
#include <string>
#include <vector>
#include "llama.h"

// Helper function to run an agent loop with memory
std::string run_agent_loop(
    LLMSession & llm,
    const std::vector<uint8_t> & cached_prompt,
    const std::string & question,
    std::vector<std::string> & memory_store,
    int max_iterations = 10
) {
    llm.load_context_from_memory(cached_prompt);

    llm += "<input>" + question + "</input>\n\n";

    std::string final_response;

    for (int i = 0; i < max_iterations; i++) {
        std::string tag = llm.select({"<think>", "<addmemory>", "<response>"});

        std::cout << "[Agent chose: " << tag << "]" << std::endl;

        if (tag == "<think>") {
            std::string thinking = llm.generate(300, {"</think>"}, 0.0f);
            std::cout << "Thinking " << (i+1) << ": " << thinking << std::endl;

        } else if (tag == "<addmemory>") {
            std::string memory_item = llm.generate(200, {"</addmemory>"}, 0.0f);
            std::cout << "Adding to memory: " << memory_item << std::endl;
            memory_store.push_back(memory_item);

        } else if (tag == "<response>") {
            final_response = llm.generate(300, {"</response>"}, 0.0f);
            std::cout << "Response: " << final_response << std::endl;
            break;
        }
    }

    if (final_response.empty()) {
        std::cout << "[Max iterations reached - forcing response]" << std::endl;
        llm += "<response>";
        final_response = llm.generate(300, {"</response>"}, 0.0f);
        std::cout << "Response: " << final_response << std::endl;
    }

    return final_response;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model-path>" << std::endl;
        return 1;
    }

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        (void) level;
        (void) text;
        (void) user_data;
    }, nullptr);

    try {
        std::cout << "=== Memory Agent Test ===" << std::endl;
        std::cout << "\nAgent can:" << std::endl;
        std::cout << "1. <think> - Reason about the problem" << std::endl;
        std::cout << "2. <addmemory> - Store important information" << std::endl;
        std::cout << "3. <response> - Provide final answer\n" << std::endl;

        LLMSession llm(argv[1], 8192);
        llm.enable_auto_cache(true);

        llm += R"(You are an AI agent with the ability to store memories.

ACTIONS YOU CAN TAKE:
- <think>reasoning</think> - Think through the problem
- <addmemory>fact</addmemory> - Store an important fact or observation
- <response>answer</response> - Provide your final response

RULES:
1. Always properly close tags
2. You can use <think> and <addmemory> multiple times
3. Use <addmemory> to store key facts, user preferences, or important observations
4. End with <response> when ready to answer
5. Think before adding to memory - only store important information

PATTERN:
<input>question</input>
<think>analyze the question</think>
<addmemory>important fact to remember</addmemory>
<think>continue reasoning</think>
<response>final answer</response>

EXAMPLES:

<input>My name is Alice and I love pizza. What's my name?</input>
<think>User is introducing themselves and asking a question</think>
<addmemory>User's name is Alice</addmemory>
<addmemory>Alice loves pizza</addmemory>
<response>Your name is Alice!</response>

<input>What is 2+2?</input>
<think>Simple arithmetic question, no need to store this</think>
<response>4</response>

<input>I'm planning a trip to Paris next month. What should I pack?</input>
<think>User is traveling to Paris</think>
<addmemory>User is planning a trip to Paris next month</addmemory>
<think>Paris weather in typical months - need to consider season</think>
<response>For Paris, pack comfortable walking shoes, layers for variable weather, a light rain jacket, and dressy casual clothes for restaurants. Don't forget a power adapter for European outlets!</response>

<input>My birthday is December 15th. Calculate how many days until New Year.</input>
<think>User's birthday is December 15th</think>
<addmemory>User's birthday is December 15th</addmemory>
<think>From Dec 15 to Dec 31 is 16 days</think>
<response>There are 16 days from December 15th to New Year's Day (January 1st).</response>

Follow this pattern. Store important user information in memory.

)";

        std::cout << "System prompt loaded and cached!" << std::endl;
        std::vector<uint8_t> cached_prompt = llm.get_cached_prompt();

        std::vector<std::string> memory_store;

        // Test 1: User introduces themselves
        std::cout << "\n=== Test 1: User Introduction ===" << std::endl;
        std::cout << "User: My name is Bob and I love hiking. What's my name?" << std::endl;
        run_agent_loop(llm, cached_prompt, "My name is Bob and I love hiking. What's my name?", memory_store);

        // Test 2: Math question (shouldn't add to memory)
        std::cout << "\n=== Test 2: Simple Question ===" << std::endl;
        std::cout << "User: What is 15 + 27?" << std::endl;
        run_agent_loop(llm, cached_prompt, "What is 15 + 27?", memory_store);

        // Test 3: User shares preferences
        std::cout << "\n=== Test 3: User Preferences ===" << std::endl;
        std::cout << "User: I'm allergic to peanuts and prefer vegetarian food. What should I order at a restaurant?" << std::endl;
        run_agent_loop(llm, cached_prompt, "I'm allergic to peanuts and prefer vegetarian food. What should I order at a restaurant?", memory_store);

        // Test 4: Complex reasoning with facts
        std::cout << "\n=== Test 4: Complex Reasoning ===" << std::endl;
        std::cout << "User: If I save $50 per week, how long until I have $1000?" << std::endl;
        run_agent_loop(llm, cached_prompt, "If I save $50 per week, how long until I have $1000?", memory_store);

        // Display memory store
        std::cout << "\n=== Memory Store Contents ===" << std::endl;
        std::cout << "Total memories stored: " << memory_store.size() << std::endl;
        for (size_t i = 0; i < memory_store.size(); i++) {
            std::cout << "  [" << (i+1) << "] " << memory_store[i] << std::endl;
        }

    } catch (const std::exception & e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
