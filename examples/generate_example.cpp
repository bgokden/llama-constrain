#include "constrained_generation.h"
#include "llama.h"
#include <iostream>
#include <cstring>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model-path>" << std::endl;
        return 1;
    }

    llama_log_set(nullptr, nullptr);

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], model_params);

    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 512;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "Failed to create context" << std::endl;
        llama_model_free(model);
        return 1;
    }

    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    std::cout << "=== Example 1: Free-form generation with max_tokens ===" << std::endl;
    {
        const char * prompt = "Once upon a time";
        std::vector<llama_token> tokens(strlen(prompt) + 8);
        int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt), tokens.data(), tokens.size(), true, false);
        tokens.resize(n_tokens);

        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size())) != 0) {
            std::cerr << "Failed to decode prompt" << std::endl;
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        generate_params params;
        params.max_tokens = 30;
        params.temperature = 0.8f;

        std::cout << "Prompt: " << prompt << std::endl;
        generate_result result = generate(ctx, vocab, params);
        std::cout << "Generated: " << result.text << std::endl;
        std::cout << "Tokens generated: " << result.tokens_generated << std::endl << std::endl;
    }

    std::cout << "=== Example 2: Generation with stop sequences ===" << std::endl;
    {
        const char * prompt = "Q: What is the capital of France?\nA:";
        std::vector<llama_token> tokens(strlen(prompt) + 16);
        int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt), tokens.data(), tokens.size(), true, false);
        tokens.resize(n_tokens);

        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size())) != 0) {
            std::cerr << "Failed to decode prompt" << std::endl;
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        generate_params params;
        params.max_tokens = 50;
        params.temperature = 0.7f;
        params.stop_sequences = {"\nQ:", "\n\n", "Question:"};

        std::cout << "Prompt: " << prompt << std::endl;
        generate_result result = generate(ctx, vocab, params);
        std::cout << "Generated: " << result.text << std::endl;
        std::cout << "Tokens generated: " << result.tokens_generated << std::endl;
        if (result.stopped_by_sequence) {
            std::cout << "Stopped by sequence: '" << result.stop_sequence << "'" << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "=== Example 3: Constrained generation with select() ===" << std::endl;
    {
        const char * prompt = "The best programming language is";
        std::vector<llama_token> tokens(strlen(prompt) + 16);
        int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt), tokens.data(), tokens.size(), true, false);
        tokens.resize(n_tokens);

        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size())) != 0) {
            std::cerr << "Failed to decode prompt" << std::endl;
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        std::vector<std::string> options = {" Python", " JavaScript", " C++", " Rust", " Go"};

        generate_params params;
        params.max_tokens = 1;
        params.temperature = 0.0f;
        params.custom_sampler = select_sampler(vocab, options);

        std::cout << "Prompt: " << prompt << std::endl;
        std::cout << "Options: ";
        for (const auto & opt : options) std::cout << opt << " ";
        std::cout << std::endl;

        generate_result result = generate(ctx, vocab, params);
        std::cout << "Selected: " << result.text << std::endl << std::endl;
    }

    std::cout << "=== Example 4: Multi-turn conversation ===" << std::endl;
    {
        const char * prompt = "Q: What's 2+2?\nA:";
        std::vector<llama_token> tokens(strlen(prompt) + 16);
        int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt), tokens.data(), tokens.size(), true, false);
        tokens.resize(n_tokens);

        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size())) != 0) {
            std::cerr << "Failed to decode prompt" << std::endl;
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        generate_params params;
        params.max_tokens = 20;
        params.temperature = 0.3f;
        params.stop_sequences = {"\nQ:"};

        std::cout << "Turn 1 - " << prompt << std::endl;
        generate_result result = generate(ctx, vocab, params);
        std::cout << "Generated: " << result.text << std::endl;
        std::cout << "Tokens: " << result.tokens_generated << std::endl << std::endl;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
