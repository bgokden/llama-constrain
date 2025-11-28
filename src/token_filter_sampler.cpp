#include "token_filter_sampler.h"
#include "constrained_llm.h"
#include "llama.h"
#include <algorithm>
#include <cstring>
#include <cctype>
#include <regex>
#include <iostream>

struct llama_sampler_token_filter {
    std::unordered_set<llama_token> token_set;
    bool is_allowlist;
};

static const char * token_filter_name(const struct llama_sampler * smpl) {
    return "token-filter";
}

static void token_filter_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_token_filter *) smpl->ctx;

    if (ctx->is_allowlist) {
        size_t write_idx = 0;
        for (size_t i = 0; i < cur_p->size; i++) {
            if (ctx->token_set.find(cur_p->data[i].id) != ctx->token_set.end()) {
                if (write_idx != i) {
                    cur_p->data[write_idx] = cur_p->data[i];
                }
                write_idx++;
            }
        }
        cur_p->size = write_idx;
    } else {
        size_t write_idx = 0;
        for (size_t i = 0; i < cur_p->size; i++) {
            if (ctx->token_set.find(cur_p->data[i].id) == ctx->token_set.end()) {
                if (write_idx != i) {
                    cur_p->data[write_idx] = cur_p->data[i];
                }
                write_idx++;
            }
        }
        cur_p->size = write_idx;
    }

    cur_p->sorted = false;
}

static void token_filter_reset(struct llama_sampler * smpl) {
}

static struct llama_sampler * token_filter_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const llama_sampler_token_filter *) smpl->ctx;
    auto * result = new llama_sampler_token_filter {
        ctx->token_set,
        ctx->is_allowlist
    };

    return llama_sampler_init(
        smpl->iface,
        result
    );
}

static void token_filter_free(struct llama_sampler * smpl) {
    delete (llama_sampler_token_filter *) smpl->ctx;
}

static struct llama_sampler_i token_filter_i = {
    /*.name   =*/ token_filter_name,
    /*.accept =*/ nullptr,
    /*.apply  =*/ token_filter_apply,
    /*.reset  =*/ token_filter_reset,
    /*.clone  =*/ token_filter_clone,
    /*.free   =*/ token_filter_free,
};

struct llama_sampler * llama_sampler_init_token_filter(
    const std::vector<llama_token> & allowed_tokens,
    bool is_allowlist
) {
    auto * ctx = new llama_sampler_token_filter {
        std::unordered_set<llama_token>(allowed_tokens.begin(), allowed_tokens.end()),
        is_allowlist
    };

    return llama_sampler_init(&token_filter_i, ctx);
}

struct llama_sampler * llama_sampler_init_token_filter_set(
    const std::unordered_set<llama_token> & token_set,
    bool is_allowlist
) {
    auto * ctx = new llama_sampler_token_filter {
        token_set,
        is_allowlist
    };

    return llama_sampler_init(&token_filter_i, ctx);
}

struct llama_sampler * llama_sampler_init_select(
    const struct llama_vocab * vocab,
    const std::vector<std::string> & options
) {
    std::unordered_set<llama_token> token_set;

    for (const auto & option : options) {
        std::vector<llama_token> tokens(option.size() + 2);
        int n = llama_tokenize(vocab, option.c_str(), option.size(), tokens.data(), tokens.size(), false, false);
        tokens.resize(n);

        if (!tokens.empty()) {
            token_set.insert(tokens[0]);
        }
    }

    auto * ctx = new llama_sampler_token_filter {
        token_set,
        true
    };

    return llama_sampler_init(&token_filter_i, ctx);
}

struct llama_sampler_prefix_select {
    std::vector<std::vector<llama_token>> option_tokens;
    std::vector<bool> active_options;
    size_t position;
};

static const char * prefix_select_name(const struct llama_sampler * smpl) {
    return "prefix-select";
}

static void prefix_select_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_prefix_select *) smpl->ctx;

    std::unordered_set<llama_token> allowed_tokens;

    for (size_t i = 0; i < ctx->option_tokens.size(); i++) {
        if (!ctx->active_options[i]) continue;

        const auto & tokens = ctx->option_tokens[i];
        if (ctx->position < tokens.size()) {
            allowed_tokens.insert(tokens[ctx->position]);
        }
    }

    // If no tokens are allowed, don't filter (avoid assert in llama.cpp)
    if (allowed_tokens.empty()) {
        return;
    }

    size_t write_idx = 0;
    for (size_t i = 0; i < cur_p->size; i++) {
        if (allowed_tokens.find(cur_p->data[i].id) != allowed_tokens.end()) {
            if (write_idx != i) {
                cur_p->data[write_idx] = cur_p->data[i];
            }
            write_idx++;
        }
    }
    cur_p->size = write_idx;
    cur_p->sorted = false;
}

static void prefix_select_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (llama_sampler_prefix_select *) smpl->ctx;

    for (size_t i = 0; i < ctx->option_tokens.size(); i++) {
        if (!ctx->active_options[i]) continue;

        const auto & tokens = ctx->option_tokens[i];
        if (ctx->position >= tokens.size() || tokens[ctx->position] != token) {
            ctx->active_options[i] = false;
        }
    }

    ctx->position++;
}

static void prefix_select_reset(struct llama_sampler * smpl) {
    auto * ctx = (llama_sampler_prefix_select *) smpl->ctx;
    std::fill(ctx->active_options.begin(), ctx->active_options.end(), true);
    ctx->position = 0;
}

static struct llama_sampler * prefix_select_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const llama_sampler_prefix_select *) smpl->ctx;
    auto * result = new llama_sampler_prefix_select {
        ctx->option_tokens,
        ctx->active_options,
        ctx->position
    };

    return llama_sampler_init(
        smpl->iface,
        result
    );
}

static void prefix_select_free(struct llama_sampler * smpl) {
    delete (llama_sampler_prefix_select *) smpl->ctx;
}

static struct llama_sampler_i prefix_select_i = {
    /*.name   =*/ prefix_select_name,
    /*.accept =*/ prefix_select_accept,
    /*.apply  =*/ prefix_select_apply,
    /*.reset  =*/ prefix_select_reset,
    /*.clone  =*/ prefix_select_clone,
    /*.free   =*/ prefix_select_free,
};

struct llama_sampler * llama_sampler_init_prefix_select(
    const struct llama_vocab * vocab,
    const std::vector<std::string> & options
) {
    std::vector<std::vector<llama_token>> option_tokens;

    for (const auto & option : options) {
        std::vector<llama_token> tokens(option.size() + 2);
        int n = llama_tokenize(vocab, option.c_str(), option.size(), tokens.data(), tokens.size(), false, false);
        tokens.resize(n);
        option_tokens.push_back(tokens);
    }

    auto * ctx = new llama_sampler_prefix_select {
        option_tokens,
        std::vector<bool>(options.size(), true),
        0
    };

    return llama_sampler_init(&prefix_select_i, ctx);
}

static bool matches_pattern(const std::string & text, PatternType pattern, const std::string & regex_pattern) {
    if (text.empty()) return false;

    switch (pattern) {
        case PATTERN_NONE:
            return true;

        case PATTERN_NUMERIC: {
            for (char c : text) {
                if (!std::isdigit(c)) return false;
            }
            return true;
        }

        case PATTERN_ALPHA: {
            for (char c : text) {
                if (!std::isalpha(c)) return false;
            }
            return true;
        }

        case PATTERN_ALPHANUMERIC: {
            for (char c : text) {
                if (!std::isalnum(c)) return false;
            }
            return true;
        }

        case PATTERN_UPPERCASE: {
            for (char c : text) {
                if (!std::isalpha(c) || !std::isupper(c)) return false;
            }
            return true;
        }

        case PATTERN_LOWERCASE: {
            for (char c : text) {
                if (!std::isalpha(c) || !std::islower(c)) return false;
            }
            return true;
        }

        case PATTERN_CAPITALIZED: {
            if (text.empty()) return false;

            size_t first_alpha_idx = 0;
            while (first_alpha_idx < text.size() && !std::isalpha(text[first_alpha_idx])) {
                first_alpha_idx++;
            }

            if (first_alpha_idx >= text.size()) return false;

            if (!std::isupper(text[first_alpha_idx])) return false;

            for (size_t i = 0; i < text.size(); i++) {
                if (!std::isalpha(text[i])) return false;
            }

            return true;
        }

        case PATTERN_REGEX:
            if (!regex_pattern.empty()) {
                try {
                    std::regex re(regex_pattern);
                    return std::regex_match(text, re);
                } catch (...) {
                    return false;
                }
            }
            return true;

        default:
            return true;
    }
}

struct llama_sampler_pattern {
    const llama_vocab * vocab;
    PatternType pattern;
    std::string regex_pattern;
    std::string accumulated;
    std::unordered_set<llama_token> stop_tokens;
};

static const char * pattern_name(const struct llama_sampler * smpl) {
    return "pattern";
}

static void pattern_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_pattern *) smpl->ctx;

    size_t write_idx = 0;
    for (size_t i = 0; i < cur_p->size; i++) {
        llama_token token = cur_p->data[i].id;

        if (ctx->stop_tokens.find(token) != ctx->stop_tokens.end()) {
            if (write_idx != i) {
                cur_p->data[write_idx] = cur_p->data[i];
            }
            write_idx++;
            continue;
        }

        char buf[256];
        int n = llama_token_to_piece(ctx->vocab, token, buf, sizeof(buf), 0, false);
        if (n > 0) {
            std::string token_text(buf, n);
            std::string test_text = ctx->accumulated + token_text;

            if (matches_pattern(test_text, ctx->pattern, ctx->regex_pattern)) {
                if (write_idx != i) {
                    cur_p->data[write_idx] = cur_p->data[i];
                }
                write_idx++;
            }
        }
    }

    cur_p->size = write_idx;
    cur_p->sorted = false;
}

static void pattern_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (llama_sampler_pattern *) smpl->ctx;

    char buf[256];
    int n = llama_token_to_piece(ctx->vocab, token, buf, sizeof(buf), 0, false);
    if (n > 0) {
        ctx->accumulated += std::string(buf, n);
    }
}

static void pattern_reset(struct llama_sampler * smpl) {
    auto * ctx = (llama_sampler_pattern *) smpl->ctx;
    ctx->accumulated.clear();
}

static struct llama_sampler * pattern_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const llama_sampler_pattern *) smpl->ctx;
    auto * result = new llama_sampler_pattern {
        ctx->vocab,
        ctx->pattern,
        ctx->regex_pattern,
        ctx->accumulated,
        ctx->stop_tokens
    };

    return llama_sampler_init(
        smpl->iface,
        result
    );
}

static void pattern_free(struct llama_sampler * smpl) {
    delete (llama_sampler_pattern *) smpl->ctx;
}

static struct llama_sampler_i pattern_i = {
    /*.name   =*/ pattern_name,
    /*.accept =*/ pattern_accept,
    /*.apply  =*/ pattern_apply,
    /*.reset  =*/ pattern_reset,
    /*.clone  =*/ pattern_clone,
    /*.free   =*/ pattern_free,
};

struct llama_sampler * llama_sampler_init_pattern(
    const struct llama_vocab * vocab,
    PatternType pattern,
    const std::string & regex_pattern,
    const std::vector<std::string> & stop_sequences
) {
    std::unordered_set<llama_token> stop_tokens;

    for (const auto & seq : stop_sequences) {
        std::vector<llama_token> tokens(seq.size() + 2);
        int n = llama_tokenize(vocab, seq.c_str(), seq.size(), tokens.data(), tokens.size(), false, false);
        tokens.resize(n);

        for (llama_token token : tokens) {
            stop_tokens.insert(token);
        }
    }

    auto * ctx = new llama_sampler_pattern {
        vocab,
        pattern,
        regex_pattern,
        "",
        stop_tokens
    };

    return llama_sampler_init(&pattern_i, ctx);
}

// Stop sequence sampler - prevents malformed tag generation
struct llama_sampler_stop_sequence {
    const llama_vocab * vocab;
    std::vector<std::string> stop_sequences;
    std::string accumulated;
};

static const char * stop_sequence_name(const struct llama_sampler * smpl) {
    return "stop-sequence";
}

static void stop_sequence_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_stop_sequence *) smpl->ctx;

    // Check if accumulated text ends with a partial stop sequence
    for (const auto & seq : ctx->stop_sequences) {
        // For sequences ending with '>', check all partial matches
        if (seq.length() > 1 && seq.back() == '>') {
            // Check progressively shorter partial matches, e.g., "</think", "</thin", "</thi", etc.
            for (size_t partial_len = seq.length() - 1; partial_len >= 2; partial_len--) {
                std::string partial = seq.substr(0, partial_len);

                if (ctx->accumulated.length() >= partial.length()) {
                    std::string acc_end = ctx->accumulated.substr(ctx->accumulated.length() - partial.length());

                    if (acc_end == partial) {
                        // Found partial match! Only allow tokens that continue toward completion
                        std::string remaining = seq.substr(partial_len);

                        // Build set of allowed tokens
                        std::unordered_set<llama_token> allowed_tokens;

                        // Allow any token that starts the remaining sequence
                        for (size_t i = 1; i <= remaining.length(); i++) {
                            std::string prefix = remaining.substr(0, i);
                            std::vector<llama_token> tokens(prefix.size() + 2);
                            int n = llama_tokenize(ctx->vocab, prefix.c_str(), prefix.size(),
                                                   tokens.data(), tokens.size(), false, false);
                            if (n > 0) {
                                allowed_tokens.insert(tokens[0]);
                            }
                        }

                        if (!allowed_tokens.empty()) {
                            // Filter to only allowed tokens
                            size_t write_idx = 0;
                            for (size_t i = 0; i < cur_p->size; i++) {
                                if (allowed_tokens.find(cur_p->data[i].id) != allowed_tokens.end()) {
                                    if (write_idx != i) {
                                        cur_p->data[write_idx] = cur_p->data[i];
                                    }
                                    write_idx++;
                                }
                            }
                            cur_p->size = write_idx;
                            cur_p->sorted = false;
                            return;
                        }
                    }
                }
            }
        }
    }
}

static void stop_sequence_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (llama_sampler_stop_sequence *) smpl->ctx;

    char buf[256];
    int n = llama_token_to_piece(ctx->vocab, token, buf, sizeof(buf), 0, false);
    if (n > 0) {
        ctx->accumulated += std::string(buf, n);
    }
}

static void stop_sequence_reset(struct llama_sampler * smpl) {
    auto * ctx = (llama_sampler_stop_sequence *) smpl->ctx;
    ctx->accumulated.clear();
}

static struct llama_sampler * stop_sequence_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const llama_sampler_stop_sequence *) smpl->ctx;
    auto * result = new llama_sampler_stop_sequence {
        ctx->vocab,
        ctx->stop_sequences,
        ctx->accumulated
    };

    return llama_sampler_init(
        smpl->iface,
        result
    );
}

static void stop_sequence_free(struct llama_sampler * smpl) {
    delete (llama_sampler_stop_sequence *) smpl->ctx;
}

static struct llama_sampler_i stop_sequence_i = {
    /*.name   =*/ stop_sequence_name,
    /*.accept =*/ stop_sequence_accept,
    /*.apply  =*/ stop_sequence_apply,
    /*.reset  =*/ stop_sequence_reset,
    /*.clone  =*/ stop_sequence_clone,
    /*.free   =*/ stop_sequence_free,
};

struct llama_sampler * llama_sampler_init_stop_sequence(
    const struct llama_vocab * vocab,
    const std::vector<std::string> & stop_sequences
) {
    auto * ctx = new llama_sampler_stop_sequence();
    ctx->vocab = vocab;
    ctx->stop_sequences = stop_sequences;
    ctx->accumulated = "";

    return llama_sampler_init(&stop_sequence_i, ctx);
}
