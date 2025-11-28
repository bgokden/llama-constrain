#ifndef GUIDANCE_H
#define GUIDANCE_H

#include "token_filter_sampler.h"
#include <string>
#include <vector>
#include <memory>
#include <map>

struct GenerateOptions {
    int min_tokens = 0;
    int max_tokens = 50;
    float temperature = 0.7f;
    std::vector<std::string> stop_sequences;
    std::string var_name;
    PatternType pattern = PATTERN_NONE;
    std::string regex_pattern;

    GenerateOptions() {}
};

class LLMSession {
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;

public:
    LLMSession(const std::string & model_path, int context_length = 2048, bool quiet = true);
    ~LLMSession();

    LLMSession(const LLMSession&) = delete;
    LLMSession& operator=(const LLMSession&) = delete;

    std::string select(const std::vector<std::string> & options, const std::string & var_name = "");

    std::string generate(int max_tokens = 50, float temperature = 0.7f, const std::string & var_name = "");

    std::string generate(
        int max_tokens,
        const std::vector<std::string> & stop_sequences,
        float temperature = 0.7f,
        const std::string & var_name = ""
    );

    std::string generate(const GenerateOptions & options);

    LLMSession& operator+=(const std::string & text);

    std::string get_output() const;

    std::string get_variable(const std::string & var_name) const;

    std::map<std::string, std::string> get_variables() const;

    void clear();

    bool save_context(const std::string & filepath) const;

    bool load_context(const std::string & filepath);

    std::vector<uint8_t> save_context_to_memory() const;

    bool load_context_from_memory(const std::vector<uint8_t> & data);

    void enable_auto_cache(bool enable = true);

    std::vector<uint8_t> get_cached_prompt() const;

    bool has_cached_prompt() const;
};

#endif
