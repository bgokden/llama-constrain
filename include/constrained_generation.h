#ifndef CONSTRAINED_GENERATION_H
#define CONSTRAINED_GENERATION_H

#include "llama.h"
#include <string>
#include <vector>
#include <functional>

struct generate_params {
    int max_tokens = 50;
    float temperature = 0.7f;
    std::vector<std::string> stop_sequences;
    llama_sampler * custom_sampler = nullptr;
};

struct generate_result {
    std::string text;
    std::vector<llama_token> tokens;
    bool stopped_by_sequence = false;
    std::string stop_sequence;
    int tokens_generated = 0;
};

generate_result generate(
    llama_context * ctx,
    const struct llama_vocab * vocab,
    const generate_params & params = generate_params()
);

llama_sampler * select_sampler(
    const struct llama_vocab * vocab,
    const std::vector<std::string> & options,
    float temperature = 0.0f
);

#endif
