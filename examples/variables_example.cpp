#include "constrained_llm.h"
#include <iostream>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model-path>" << std::endl;
        return 1;
    }

    try {
        std::cout << "=== Example 1: Custom Context Length ===" << std::endl;
        LLMSession llm(argv[1], 512);

        llm += "The capital of France is ";
        std::string city = llm.select({"Paris", "London", "Berlin"}, "city");
        std::cout << "City variable: '" << llm.get_variable("city") << "'" << std::endl;
        std::cout << "Full output: " << llm.get_output() << std::endl << std::endl;

        llm.clear();

        std::cout << "=== Example 2: Variable Extraction ===" << std::endl;
        llm += "Q: What is your favorite color?\nA: ";
        std::string color = llm.select({"red", "blue", "green"}, "color");

        llm += "\n\nQ: What is your favorite animal?\nA: ";
        std::string animal = llm.generate(10, {"\n"}, 0.7f, "animal");

        std::cout << "Color: '" << llm.get_variable("color") << "'" << std::endl;
        std::cout << "Animal: '" << llm.get_variable("animal") << "'" << std::endl << std::endl;

        llm.clear();

        std::cout << "=== Example 3: Multiple Variables ===" << std::endl;
        llm += "Name: ";
        llm.generate(5, {"\n"}, 0.5f, "name");

        llm += "\nAge: ";
        llm.generate(3, {"\n"}, 0.0f, "age");

        llm += "\nCity: ";
        llm.select({"Paris", "London", "Tokyo", "New York"}, "city");

        std::cout << "All variables:" << std::endl;
        auto vars = llm.get_variables();
        for (const auto & pair : vars) {
            std::cout << "  " << pair.first << " = '" << pair.second << "'" << std::endl;
        }
        std::cout << std::endl;

        llm.clear();

        std::cout << "=== Example 4: Building Structured Data ===" << std::endl;
        llm += "Product: Laptop\nPrice: $";
        llm.generate(5, {"\n"}, 0.3f, "price");

        llm += "\nRating: ";
        llm.select({"1 star", "2 stars", "3 stars", "4 stars", " 5 stars"}, "rating");

        llm += "\nIn stock: ";
        llm.select({"Yes", "No"}, "in_stock");

        std::cout << "Product data:" << std::endl;
        std::cout << "  Price: '" << llm.get_variable("price") << "'" << std::endl;
        std::cout << "  Rating: '" << llm.get_variable("rating") << "'" << std::endl;
        std::cout << "  In Stock: '" << llm.get_variable("in_stock") << "'" << std::endl;

    } catch (const std::exception & e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
