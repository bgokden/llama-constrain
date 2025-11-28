#ifndef TOKEN_FILTER_SAMPLER_H
#define TOKEN_FILTER_SAMPLER_H

#include "llama.h"
#include <vector>
#include <unordered_set>
#include <string>

enum PatternType {
    PATTERN_NONE = 0,
    PATTERN_NUMERIC,
    PATTERN_ALPHA,
    PATTERN_ALPHANUMERIC,
    PATTERN_UPPERCASE,
    PATTERN_LOWERCASE,
    PATTERN_CAPITALIZED,
    PATTERN_REGEX
};

struct llama_sampler * llama_sampler_init_token_filter(
    const std::vector<llama_token> & allowed_tokens,
    bool is_allowlist = true
);

struct llama_sampler * llama_sampler_init_token_filter_set(
    const std::unordered_set<llama_token> & token_set,
    bool is_allowlist = true
);

struct llama_sampler * llama_sampler_init_select(
    const struct llama_vocab * vocab,
    const std::vector<std::string> & options
);

struct llama_sampler * llama_sampler_init_prefix_select(
    const struct llama_vocab * vocab,
    const std::vector<std::string> & options
);

struct llama_sampler * llama_sampler_init_pattern(
    const struct llama_vocab * vocab,
    PatternType pattern,
    const std::string & regex_pattern = "",
    const std::vector<std::string> & stop_sequences = std::vector<std::string>()
);

struct llama_sampler * llama_sampler_init_stop_sequence(
    const struct llama_vocab * vocab,
    const std::vector<std::string> & stop_sequences
);

#endif
