#include "token_filter_sampler.h"
#include "llama.h"
#include <iostream>
#include <vector>
#include <cstring>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model-path>" << std::endl;
        return 1;
    }

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

    const char * prompt = "The capital of France is";

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(strlen(prompt) + 8);
    int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt), tokens.data(), tokens.size(), true, false);
    tokens.resize(n_tokens);

    std::cout << "Prompt: " << prompt << std::endl;
    std::cout << "Tokenized to " << tokens.size() << " tokens" << std::endl;

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size())) != 0) {
        std::cerr << "Failed to decode prompt" << std::endl;
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    std::vector<llama_token> allowed_tokens;

    std::vector<std::string> allowed_words = {" Paris", " London", " Berlin", " Madrid", " Rome"};
    for (const auto & word : allowed_words) {
        std::vector<llama_token> word_tokens(word.size() + 2);
        int n = llama_tokenize(vocab, word.c_str(), word.size(), word_tokens.data(), word_tokens.size(), false, false);
        word_tokens.resize(n);
        if (!word_tokens.empty()) {
            allowed_tokens.push_back(word_tokens[0]);
            std::cout << "Allowed: " << word << " (token " << word_tokens[0] << ")" << std::endl;
        }
    }

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_token_filter(allowed_tokens, true));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));

    std::cout << "\nGenerating with strict token filter..." << std::endl;
    std::cout << "Output: " << prompt;

    for (int i = 0; i < 10; i++) {
        llama_token new_token = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        char buf[256];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, false);
        if (n > 0) {
            std::cout << std::string(buf, n);
            std::cout.flush();
        }

        tokens.push_back(new_token);

        if (llama_decode(ctx, llama_batch_get_one(&new_token, 1)) != 0) {
            std::cerr << "\nFailed to decode token" << std::endl;
            break;
        }
    }

    std::cout << std::endl;

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
