#include "constrained_llm.h"
#include "constrained_generation.h"
#include "llama.h"
#include <iostream>
#include <sstream>
#include <map>

struct LLMSession::Impl {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    std::string accumulated_text;
    std::vector<llama_token> context_tokens;
    std::map<std::string, std::string> variables;
    bool auto_cache_enabled = false;
    std::vector<uint8_t> cached_prompt_data;
    bool has_cached = false;

    ~Impl() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
        llama_backend_free();
    }

    void encode_and_eval(const std::string & text) {
        std::vector<llama_token> tokens(text.size() + 16);
        int n = llama_tokenize(vocab, text.c_str(), text.size(),
                               tokens.data(), tokens.size(),
                               context_tokens.empty(), false);
        tokens.resize(n);

        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size())) != 0) {
            throw std::runtime_error("Failed to decode text");
        }

        context_tokens.insert(context_tokens.end(), tokens.begin(), tokens.end());
    }
};

LLMSession::LLMSession(const std::string & model_path, int context_length, bool quiet)
    : pImpl(new Impl()) {

    if (quiet) {
        llama_log_set(nullptr, nullptr);
    }

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    pImpl->model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (!pImpl->model) {
        throw std::runtime_error("Failed to load model from: " + model_path);
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = context_length;
    ctx_params.n_batch = context_length > 2048 ? 2048 : context_length;

    pImpl->ctx = llama_init_from_model(pImpl->model, ctx_params);
    if (!pImpl->ctx) {
        throw std::runtime_error("Failed to create context");
    }

    pImpl->vocab = llama_model_get_vocab(pImpl->model);
}

LLMSession::~LLMSession() = default;

std::string LLMSession::select(const std::vector<std::string> & options, const std::string & var_name) {
    // Tokenize all options
    std::vector<std::vector<llama_token>> option_tokens;
    int max_length = 0;

    for (const auto & opt : options) {
        std::vector<llama_token> tokens(opt.size() + 8);
        int n = llama_tokenize(pImpl->vocab, opt.c_str(), opt.size(), tokens.data(), tokens.size(), false, false);
        tokens.resize(n);
        option_tokens.push_back(tokens);
        max_length = std::max(max_length, n);
    }

    // Generate with prefix_select sampler, checking after each token if we've matched an option
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_prefix_select(pImpl->vocab, options));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));

    std::vector<llama_token> generated_tokens;
    std::string selected;

    for (int i = 0; i < max_length; i++) {
        llama_token new_token = llama_sampler_sample(smpl, pImpl->ctx, -1);

        if (llama_vocab_is_eog(pImpl->vocab, new_token)) {
            break;
        }

        generated_tokens.push_back(new_token);

        // Check if we've fully matched any option
        for (size_t opt_idx = 0; opt_idx < option_tokens.size(); opt_idx++) {
            if (generated_tokens.size() == option_tokens[opt_idx].size()) {
                bool match = true;
                for (size_t j = 0; j < generated_tokens.size(); j++) {
                    if (generated_tokens[j] != option_tokens[opt_idx][j]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    selected = options[opt_idx];
                    break;
                }
            }
        }

        // If we found a match, stop early
        if (!selected.empty()) {
            break;
        }

        // Decode token into context
        if (llama_decode(pImpl->ctx, llama_batch_get_one(&new_token, 1)) != 0) {
            throw std::runtime_error("Failed to decode token");
        }
    }

    llama_sampler_free(smpl);

    // Add tokens to context
    for (llama_token token : generated_tokens) {
        pImpl->context_tokens.push_back(token);
    }

    pImpl->accumulated_text += selected;

    if (!var_name.empty()) {
        pImpl->variables[var_name] = selected;
    }

    return selected;
}

std::string LLMSession::generate(int max_tokens, float temperature, const std::string & var_name) {
    return generate(max_tokens, {}, temperature, var_name);
}

std::string LLMSession::generate(
    int max_tokens,
    const std::vector<std::string> & stop_sequences,
    float temperature,
    const std::string & var_name
) {
    generate_params params;
    params.max_tokens = max_tokens;
    params.temperature = temperature;
    params.stop_sequences = stop_sequences;

    generate_result result = ::generate(pImpl->ctx, pImpl->vocab, params);

    // Tokens in result.tokens were already decoded to llama context during generation
    // (except possibly the last token if stopped by sequence)
    // We just need to track them
    for (llama_token token : result.tokens) {
        pImpl->context_tokens.push_back(token);
    }

    pImpl->accumulated_text += result.text;

    // If generation stopped due to a stop sequence, encode and add it to context
    if (result.stopped_by_sequence && !result.stop_sequence.empty()) {
        pImpl->encode_and_eval(result.stop_sequence);
        pImpl->accumulated_text += result.stop_sequence;
    }
    // If we didn't stop by sequence but have stop sequences defined,
    // check if we should auto-complete a stop sequence
    else if (!stop_sequences.empty() && result.tokens_generated >= params.max_tokens) {
        // Check if the generated text ends with a partial stop sequence
        bool completed = false;
        for (const auto & seq : stop_sequences) {
            if (completed) break;

            // Check if the end of generated text starts any of the stop sequences
            // Start from longest match to shortest
            for (int i = seq.length() - 1; i >= 1; i--) {
                if (result.text.length() >= (size_t)i) {
                    std::string end_part = result.text.substr(result.text.length() - i);
                    std::string seq_start = seq.substr(0, i);

                    if (end_part == seq_start) {
                        // Found partial match, complete the stop sequence
                        std::string remainder = seq.substr(i);
                        std::cerr << "[AUTO-COMPLETE] Text ends with: '" << end_part << "'" << std::endl;
                        std::cerr << "[AUTO-COMPLETE] Completing with: '" << remainder << "'" << std::endl;
                        pImpl->encode_and_eval(remainder);
                        pImpl->accumulated_text += remainder;
                        completed = true;
                        break;
                    }
                }
            }
        }
    }

    if (!var_name.empty()) {
        pImpl->variables[var_name] = result.text;
    }

    return result.text;
}

std::string LLMSession::generate(const GenerateOptions & options) {
    generate_params params;
    params.max_tokens = options.max_tokens;
    params.temperature = options.temperature;
    params.stop_sequences = options.stop_sequences;

    if (options.pattern != PATTERN_NONE) {
        params.custom_sampler = llama_sampler_init_pattern(
            pImpl->vocab,
            options.pattern,
            options.regex_pattern,
            options.stop_sequences
        );
    }

    generate_result result = ::generate(pImpl->ctx, pImpl->vocab, params);

    if (result.tokens_generated < options.min_tokens && !result.stopped_by_sequence) {
        int additional_tokens = options.min_tokens - result.tokens_generated;
        params.max_tokens = additional_tokens;
        params.stop_sequences.clear();

        generate_result additional = ::generate(pImpl->ctx, pImpl->vocab, params);

        for (llama_token token : additional.tokens) {
            pImpl->context_tokens.push_back(token);
        }
        result.text += additional.text;
        result.tokens_generated += additional.tokens_generated;
    }

    for (llama_token token : result.tokens) {
        pImpl->context_tokens.push_back(token);
    }

    pImpl->accumulated_text += result.text;

    // If generation stopped due to a stop sequence, add it to the context
    if (result.stopped_by_sequence && !result.stop_sequence.empty()) {
        pImpl->encode_and_eval(result.stop_sequence);
        pImpl->accumulated_text += result.stop_sequence;
    }
    // If we didn't stop by sequence but have stop sequences defined,
    // check if we should auto-complete a stop sequence
    else if (!options.stop_sequences.empty() && result.tokens_generated >= params.max_tokens) {
        // Check if the generated text ends with a partial stop sequence
        bool completed = false;
        for (const auto & seq : options.stop_sequences) {
            if (completed) break;

            // Check if the end of generated text starts any of the stop sequences
            // Start from longest match to shortest
            for (int i = seq.length() - 1; i >= 1; i--) {
                if (result.text.length() >= (size_t)i) {
                    std::string end_part = result.text.substr(result.text.length() - i);
                    std::string seq_start = seq.substr(0, i);

                    if (end_part == seq_start) {
                        // Found partial match, complete the stop sequence
                        std::string remainder = seq.substr(i);
                        std::cerr << "[AUTO-COMPLETE] Text ends with: '" << end_part << "'" << std::endl;
                        std::cerr << "[AUTO-COMPLETE] Completing with: '" << remainder << "'" << std::endl;
                        pImpl->encode_and_eval(remainder);
                        pImpl->accumulated_text += remainder;
                        completed = true;
                        break;
                    }
                }
            }
        }
    }

    if (!options.var_name.empty()) {
        pImpl->variables[options.var_name] = result.text;
    }

    return result.text;
}

LLMSession& LLMSession::operator+=(const std::string & text) {
    pImpl->encode_and_eval(text);
    pImpl->accumulated_text += text;

    if (pImpl->auto_cache_enabled && !pImpl->has_cached && !pImpl->context_tokens.empty()) {
        pImpl->cached_prompt_data = save_context_to_memory();
        pImpl->has_cached = true;
    }

    return *this;
}

std::string LLMSession::get_output() const {
    return pImpl->accumulated_text;
}

std::string LLMSession::get_variable(const std::string & var_name) const {
    auto it = pImpl->variables.find(var_name);
    if (it != pImpl->variables.end()) {
        return it->second;
    }
    return "";
}

std::map<std::string, std::string> LLMSession::get_variables() const {
    return pImpl->variables;
}

void LLMSession::clear() {
    pImpl->accumulated_text.clear();
    pImpl->context_tokens.clear();
    pImpl->variables.clear();
}

bool LLMSession::save_context(const std::string & filepath) const {
    size_t state_size = llama_state_get_size(pImpl->ctx);
    std::vector<uint8_t> state_data(state_size);

    size_t written = llama_state_get_data(pImpl->ctx, state_data.data(), state_size);
    if (written == 0) {
        return false;
    }

    FILE * fp = fopen(filepath.c_str(), "wb");
    if (!fp) {
        return false;
    }

    size_t tokens_count = pImpl->context_tokens.size();
    fwrite(&tokens_count, sizeof(size_t), 1, fp);
    if (tokens_count > 0) {
        fwrite(pImpl->context_tokens.data(), sizeof(llama_token), tokens_count, fp);
    }

    size_t text_size = pImpl->accumulated_text.size();
    fwrite(&text_size, sizeof(size_t), 1, fp);
    if (text_size > 0) {
        fwrite(pImpl->accumulated_text.c_str(), sizeof(char), text_size, fp);
    }

    fwrite(&written, sizeof(size_t), 1, fp);
    fwrite(state_data.data(), 1, written, fp);

    fclose(fp);
    return true;
}

bool LLMSession::load_context(const std::string & filepath) {
    FILE * fp = fopen(filepath.c_str(), "rb");
    if (!fp) {
        return false;
    }

    size_t tokens_count = 0;
    fread(&tokens_count, sizeof(size_t), 1, fp);
    pImpl->context_tokens.resize(tokens_count);
    if (tokens_count > 0) {
        fread(pImpl->context_tokens.data(), sizeof(llama_token), tokens_count, fp);
    }

    size_t text_size = 0;
    fread(&text_size, sizeof(size_t), 1, fp);
    pImpl->accumulated_text.resize(text_size);
    if (text_size > 0) {
        fread(&pImpl->accumulated_text[0], sizeof(char), text_size, fp);
    }

    size_t state_size = 0;
    fread(&state_size, sizeof(size_t), 1, fp);

    std::vector<uint8_t> state_data(state_size);
    fread(state_data.data(), 1, state_size, fp);

    size_t loaded = llama_state_set_data(pImpl->ctx, state_data.data(), state_size);
    if (loaded == 0) {
        fclose(fp);
        return false;
    }

    fclose(fp);
    return true;
}

std::vector<uint8_t> LLMSession::save_context_to_memory() const {
    std::vector<uint8_t> buffer;

    size_t state_size = llama_state_get_size(pImpl->ctx);
    std::vector<uint8_t> state_data(state_size);

    size_t written = llama_state_get_data(pImpl->ctx, state_data.data(), state_size);
    if (written == 0) {
        return buffer;
    }

    size_t tokens_count = pImpl->context_tokens.size();
    size_t text_size = pImpl->accumulated_text.size();

    size_t total_size = sizeof(size_t) + tokens_count * sizeof(llama_token) +
                        sizeof(size_t) + text_size +
                        sizeof(size_t) + written;

    buffer.reserve(total_size);

    auto write_data = [&buffer](const void* data, size_t size) {
        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        buffer.insert(buffer.end(), bytes, bytes + size);
    };

    write_data(&tokens_count, sizeof(size_t));
    if (tokens_count > 0) {
        write_data(pImpl->context_tokens.data(), tokens_count * sizeof(llama_token));
    }

    write_data(&text_size, sizeof(size_t));
    if (text_size > 0) {
        write_data(pImpl->accumulated_text.c_str(), text_size);
    }

    write_data(&written, sizeof(size_t));
    write_data(state_data.data(), written);

    return buffer;
}

bool LLMSession::load_context_from_memory(const std::vector<uint8_t> & data) {
    if (data.empty()) {
        return false;
    }

    size_t offset = 0;

    auto read_data = [&data, &offset](void* dest, size_t size) -> bool {
        if (offset + size > data.size()) {
            return false;
        }
        std::memcpy(dest, data.data() + offset, size);
        offset += size;
        return true;
    };

    size_t tokens_count = 0;
    if (!read_data(&tokens_count, sizeof(size_t))) {
        return false;
    }

    pImpl->context_tokens.resize(tokens_count);
    if (tokens_count > 0) {
        if (!read_data(pImpl->context_tokens.data(), tokens_count * sizeof(llama_token))) {
            return false;
        }
    }

    size_t text_size = 0;
    if (!read_data(&text_size, sizeof(size_t))) {
        return false;
    }

    pImpl->accumulated_text.resize(text_size);
    if (text_size > 0) {
        if (!read_data(&pImpl->accumulated_text[0], text_size)) {
            return false;
        }
    }

    size_t state_size = 0;
    if (!read_data(&state_size, sizeof(size_t))) {
        return false;
    }

    if (offset + state_size != data.size()) {
        return false;
    }

    size_t loaded = llama_state_set_data(pImpl->ctx, data.data() + offset, state_size);
    if (loaded == 0) {
        return false;
    }

    return true;
}

void LLMSession::enable_auto_cache(bool enable) {
    pImpl->auto_cache_enabled = enable;
}

std::vector<uint8_t> LLMSession::get_cached_prompt() const {
    return pImpl->cached_prompt_data;
}

bool LLMSession::has_cached_prompt() const {
    return pImpl->has_cached;
}
