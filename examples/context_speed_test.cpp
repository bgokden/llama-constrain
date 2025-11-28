#include "constrained_llm.h"
#include <iostream>
#include <chrono>
#include <cstdio>

using namespace std::chrono;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model-path>" << std::endl;
        return 1;
    }

    const char * context_file = "speed_test_context.bin";

    std::string long_prompt = R"(You are a helpful assistant that extracts structured data.

Example 1:
Input: John Smith, age 42, lives in Paris, works as Engineer
Output: Name=John Smith, Age=42, City=Paris, Occupation=Engineer

Example 2:
Input: Sarah Johnson, age 35, lives in London, works as Doctor
Output: Name=Sarah Johnson, Age=35, City=London, Occupation=Doctor

Example 3:
Input: Michael Chen, age 28, lives in Tokyo, works as Designer
Output: Name=Michael Chen, Age=28, City=Tokyo, Occupation=Designer

Example 4:
Input: Emma Wilson, age 31, lives in Berlin, works as Teacher
Output: Name=Emma Wilson, Age=31, City=Berlin, Occupation=Teacher

Example 5:
Input: David Martinez, age 45, lives in Madrid, works as Lawyer
Output: Name=David Martinez, Age=45, City=Madrid, Occupation=Lawyer

Example 6:
Input: Lisa Anderson, age 39, lives in New York, works as Architect
Output: Name=Lisa Anderson, Age=39, City=New York, Occupation=Architect

Example 7:
Input: Robert Taylor, age 33, lives in Sydney, works as Chef
Output: Name=Robert Taylor, Age=33, City=Sydney, Occupation=Chef

Example 8:
Input: Maria Garcia, age 27, lives in Barcelona, works as Journalist
Output: Name=Maria Garcia, Age=27, City=Barcelona, Occupation=Journalist

Example 9:
Input: James Brown, age 50, lives in Chicago, works as Pilot
Output: Name=James Brown, Age=50, City=Chicago, Occupation=Pilot

Example 10:
Input: Sophie Dubois, age 29, lives in Montreal, works as Scientist
Output: Name=Sophie Dubois, Age=29, City=Montreal, Occupation=Scientist

Now extract data from the following:
)";

    try {
        std::cout << "=== Context Loading Speed Test ===" << std::endl;
        std::cout << "Long prompt length: " << long_prompt.length() << " characters\n" << std::endl;

        // Test 1: Process prompt directly (without save/load)
        std::cout << "Test 1: Processing prompt directly..." << std::endl;
        auto start1 = high_resolution_clock::now();

        LLMSession llm1(argv[1], 4096);
        llm1 += long_prompt;

        auto end1 = high_resolution_clock::now();
        auto duration1 = duration_cast<milliseconds>(end1 - start1);
        std::cout << "Time to process prompt directly: " << duration1.count() << " ms\n" << std::endl;

        // Test 2: Save context
        std::cout << "Test 2: Saving context to file..." << std::endl;
        auto start_save = high_resolution_clock::now();

        if (!llm1.save_context(context_file)) {
            std::cerr << "Failed to save context!" << std::endl;
            return 1;
        }

        auto end_save = high_resolution_clock::now();
        auto duration_save = duration_cast<milliseconds>(end_save - start_save);
        std::cout << "Time to save context: " << duration_save.count() << " ms\n" << std::endl;

        // Test 3: Load context
        std::cout << "Test 3: Loading context from file..." << std::endl;
        auto start_load = high_resolution_clock::now();

        LLMSession llm2(argv[1], 4096);
        if (!llm2.load_context(context_file)) {
            std::cerr << "Failed to load context!" << std::endl;
            return 1;
        }

        auto end_load = high_resolution_clock::now();
        auto duration_load = duration_cast<milliseconds>(end_load - start_load);
        std::cout << "Time to load context: " << duration_load.count() << " ms\n" << std::endl;

        // Test 4: Process same prompt again directly (for comparison)
        std::cout << "Test 4: Processing same prompt again directly..." << std::endl;
        auto start2 = high_resolution_clock::now();

        LLMSession llm3(argv[1], 4096);
        llm3 += long_prompt;

        auto end2 = high_resolution_clock::now();
        auto duration2 = duration_cast<milliseconds>(end2 - start2);
        std::cout << "Time to process prompt directly (2nd run): " << duration2.count() << " ms\n" << std::endl;

        // Summary
        std::cout << "=== Speed Comparison Summary ===" << std::endl;
        std::cout << "Direct prompt processing:  " << duration1.count() << " ms (average)" << std::endl;
        std::cout << "Load from saved context:   " << duration_load.count() << " ms" << std::endl;
        std::cout << "Save context to file:      " << duration_save.count() << " ms" << std::endl;

        float speedup = static_cast<float>(duration1.count()) / duration_load.count();
        std::cout << "\nSpeedup: " << speedup << "x faster to load from file" << std::endl;

        long total_time_with_save_load = duration_save.count() + duration_load.count();
        std::cout << "Total time (save + load):  " << total_time_with_save_load << " ms" << std::endl;

        if (total_time_with_save_load < duration1.count()) {
            std::cout << "Note: Even with save overhead, loading is faster!" << std::endl;
        } else {
            std::cout << "Note: Savings appear after 2+ reuses of saved context" << std::endl;
        }

        // Verify both contexts produce same output
        std::cout << "\n=== Verification: Both contexts should produce similar output ===" << std::endl;

        std::cout << "Generating from directly loaded prompt..." << std::endl;
        llm1 += "Input: Test Person, age 40, lives in Boston, works as Developer\nOutput: ";
        std::string output1 = llm1.generate(30, {"\n"}, 0.3f);
        std::cout << "Output 1: " << output1 << std::endl;

        std::cout << "\nGenerating from loaded context..." << std::endl;
        llm2 += "Input: Test Person, age 40, lives in Boston, works as Developer\nOutput: ";
        std::string output2 = llm2.generate(30, {"\n"}, 0.3f);
        std::cout << "Output 2: " << output2 << std::endl;

        // Clean up
        std::remove(context_file);

    } catch (const std::exception & e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
