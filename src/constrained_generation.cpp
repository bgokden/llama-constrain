#include "constrained_generation.h"
#include "token_filter_sampler.h"
#include <iostream>

static bool check_stop_sequence(
    const std::string & generated_text,
    const std::vector<std::string> & stop_sequences,
    std::string & found_sequence
) {
    for (const auto & seq : stop_sequences) {
        if (generated_text.find(seq) != std::string::npos) {
            found_sequence = seq;
            return true;
        }
    }
    return false;
}

generate_result generate(
    llama_context * ctx,
    const struct llama_vocab * vocab,
    const generate_params & params
) {
    generate_result result;

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    if (params.custom_sampler) {
        llama_sampler_chain_add(smpl, params.custom_sampler);
    }

    // Add stop sequence sampler if stop sequences are provided
    if (!params.stop_sequences.empty()) {
        llama_sampler_chain_add(smpl, llama_sampler_init_stop_sequence(vocab, params.stop_sequences));
    }

    llama_sampler_chain_add(smpl, llama_sampler_init_temp(params.temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));

    for (int i = 0; i < params.max_tokens; i++) {
        llama_token new_token = llama_sampler_sample(smpl, ctx, -1);
        // Note: llama_sampler_sample() already calls llama_sampler_accept() internally

        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        char buf[256];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, false);
        if (n > 0) {
            std::string token_str(buf, n);

            result.text += token_str;
            result.tokens.push_back(new_token);
            result.tokens_generated++;

            // Check for complete stop sequence
            if (!params.stop_sequences.empty()) {
                std::string found_seq;
                if (check_stop_sequence(result.text, params.stop_sequences, found_seq)) {
                    result.stopped_by_sequence = true;
                    result.stop_sequence = found_seq;

                    // Remove stop sequence from returned text
                    size_t pos = result.text.find(found_seq);
                    if (pos != std::string::npos) {
                        result.text = result.text.substr(0, pos);
                    }
                    break;
                }
            }

            // Decode token into context
            if (llama_decode(ctx, llama_batch_get_one(&new_token, 1)) != 0) {
                std::cerr << "Failed to decode token" << std::endl;
                break;
            }
        }
    }

    llama_sampler_free(smpl);
    return result;
}

llama_sampler * select_sampler(
    const struct llama_vocab * vocab,
    const std::vector<std::string> & options,
    float temperature
) {
    return llama_sampler_init_prefix_select(vocab, options);
}
