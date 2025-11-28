# llama-constrain

A powerful **constrained generation** library for llama.cpp with an ultra-simple API. Control exactly what tokens your model can generate - force specific outputs, ensure valid JSON/XML, or constrain to patterns. No manual tokenization, no decoding, no memory management required!

## 🚀 Why llama-constrain?

**Constrained generation** means controlling which tokens the model can generate at each step. This library provides the tools to:

- **Force specific choices** - Make model select from predefined options (e.g., `select(["Yes", "No", "Maybe"])`)
- **Ensure valid structure** - Generate well-formed XML/JSON with proper tag completion
- **Pattern matching** - Constrain output to numbers, emails, or custom regex patterns
- **Stop sequences** - Reliably stop at specific strings with proper completion

All built on llama.cpp's sampler architecture using **pre-generation token filtering** - tokens are filtered before selection, ensuring constraints are always satisfied.

### Bonus: Multi-Step Thinking

As an example of what constrained generation enables, you can easily implement **chain-of-thought reasoning** where any model "thinks" step-by-step using structured tags, even without specific training!

## Quick Examples

```cpp
// 1. Force model to choose from specific options
std::string answer = llm.select({"Yes", "No", "Maybe"});

// 2. Generate valid JSON with proper closing
std::string json = llm.generate(100, {"\n}"}, 0.0f);

// 3. Constrain to numbers only
llm.generate(10, {}, 0.7f, "", PATTERN_NUMERIC);

// 4. Make any model think step-by-step
std::string thinking = llm.generate(300, {"</think>"}, 0.0f);
```

## ✨ Features

### 🎯 High-Level APIs

- **`select()`** - Choose from predefined options (forced choice)
- **`generate()`** - Free-form generation with smart constraints
  - `max_tokens` - Limit generation length
  - `stop_sequences` - Stop at specific strings (with proper XML/JSON completion)
  - `temperature` - Control randomness (including temperature 0 for deterministic output)
  - `custom_sampler` - Add custom constraints

### 🛡️ Advanced Samplers

- **Stop Sequence Sampler** - Prevents malformed XML/JSON tags
  - Automatically completes partial stop sequences (e.g., `</think` → `</think>`)
  - Blocks tokens that would create malformed patterns
  - Works with any XML-style structured output

- **Prefix Select Sampler** - Multi-token option selection
  - Forces exact string matches from a list of options
  - Handles multi-token options correctly

- **Pattern Sampler** - Constrain output format
  - Numeric, alphabetic, alphanumeric patterns
  - Custom regex patterns
  - Per-token validation

### ⚡ Performance Optimizations

- **Automatic Context Caching** - Save and reuse prompt processing
  - Cache large system prompts (KV cache state + tokens + text)
  - Restore context for repeated queries without reprocessing
  - Perfect for static prompts with multiple questions
  - Significant speedup: ~600 token prompt processed once, reused N times

### 🔧 Low-Level Token Filtering

- **Allowlist mode**: Only specified tokens can be generated
- **Blocklist mode**: All tokens except specified ones can be generated
- **Efficient filtering**: Modifies token candidate array in-place before sampling
- **Modular design**: Integrates seamlessly with llama.cpp's sampler chain

## Implementation

The sampler implements the `llama_sampler_i` interface with:

- `apply()`: Filters tokens based on allowlist/blocklist
- `clone()`: Creates independent copies of the sampler
- `reset()`: No-op (stateless sampler)
- `free()`: Proper cleanup

## Project Structure

```
custom-sampler/
├── include/
│   └── token_filter_sampler.h
├── src/
│   └── token_filter_sampler.cpp
├── examples/
│   └── example.cpp
├── CMakeLists.txt
└── README.md
```

## Building

### Quick Start (Using Makefile)

```bash
# First time setup - build llama.cpp
make llama

# Build the custom sampler
make build

# Run the example (provide your model path)
make run MODEL_PATH=/path/to/your/model.gguf
```

### Manual Build (Using CMake)

1. Initialize and build llama.cpp submodule:
```bash
git submodule update --init --recursive
cd external/llama.cpp
cmake -B build
cmake --build build -j4
cd ../..
```

2. Build this project:
```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### Makefile Commands

- `make llama` - Build llama.cpp (required on first run)
- `make build` - Build the custom sampler library and example
- `make run MODEL_PATH=/path/to/model.gguf` - Run the example
- `make clean` - Remove build artifacts
- `make clean-all` - Remove all builds including llama.cpp
- `make help` - Show available commands

## Usage

### Ultra-Simple API (Recommended)

The easiest way to use constrained generation - no manual tokenization or memory management!

```cpp
#include "constrained_llm.h"

// Create session - handles everything automatically!
LLMSession llm("model.gguf");

// Example 1: Select from options
llm += "The capital of France is";
std::string city = llm.select({" Paris", " London", " Berlin", " Madrid"});
// Output: "The capital of France is Paris"

// Example 2: Generate with stop sequences
llm.clear();
llm += "Q: What is 2+2?\nA:";
std::string answer = llm.generate(30, {"\nQ:", "\n\n"});

// Example 3: Chain operations
llm.clear();
llm += "Story: Once upon a time";
llm.generate(50);  // Generate 50 tokens
llm += "\n\nGenre:";
std::string genre = llm.select({" Fantasy", " Sci-Fi", " Mystery"});

std::cout << llm.get_output();  // Get full conversation
```

**That's it!** No tokenization, no `llama_decode()`, no cleanup - everything is automatic.

### Example: Structured Thinking (Chain-of-Thought)

Using constrained generation, you can make **any model** perform multi-step reasoning with structured XML outputs - no special training required:

```cpp
#include "constrained_llm.h"

LLMSession llm("model.gguf");

// System prompt for structured thinking
llm += R"(You are a helpful assistant that thinks step-by-step.
Use <think> tags for reasoning and <output> tags for final answers.
Example: <think>First I need to...</think><output>The answer is...</output>
)";

// Question
llm += "<input>What is 123 + 456?</input>\n\n";

// Thinking loop with minimum 2 steps
for (int i = 0; i < 5; i++) {
    std::string choice;
    if (i < 2) {
        choice = "<think>";
        llm += choice;
    } else {
        choice = llm.select({"<think>", "<output>"});
    }

    if (choice == "<think>") {
        std::string thinking = llm.generate(300, {"</think>"}, 0.0f);
        std::cout << "Thinking " << (i+1) << ": " << thinking << std::endl;
    } else if (choice == "<output>") {
        std::string answer = llm.generate(200, {"</output>"}, 0.0f);
        std::cout << "Output: " << answer << std::endl;
        break;
    }
}
```

Output:
```
Thinking 1: Break down the addition into hundreds, tens, and ones:
Thinking 2: 100 + 400 = 500
Thinking 3: 20 + 50 = 70
Thinking 4: 3 + 6 = 9
Thinking 5: Combine the results: 500 + 70 + 9 = 579
Output: 579
```

**Key Feature**: The stop sequence sampler automatically prevents malformed tags like `</think</think</think` by filtering tokens before they're generated!

### Context Caching for Repeated Queries

When you have a static system prompt and multiple queries, cache the prompt to avoid reprocessing:

```cpp
LLMSession llm("model.gguf");
llm.enable_auto_cache(true);  // Enable automatic caching

// Large system prompt (processed once)
llm += "You are a helpful assistant that...[long prompt with examples]...";

// Save the cached state (KV cache + tokens + text)
std::vector<uint8_t> cached_prompt = llm.get_cached_prompt();

// Process multiple queries - each starts from cached state
for (const auto& question : questions) {
    llm.load_context_from_memory(cached_prompt);  // Restore to system prompt
    llm += "<input>" + question + "</input>";
    std::string answer = llm.generate(200, {"</output>"});
    // Next iteration resets back to just the system prompt
}
```

**Performance**: A 600-token system prompt is processed once and reused for all queries, saving significant computation time.

### Mid-Level API (More Control)

#### 1. `select()` - Choose from Options

```cpp
#include "constrained_generation.h"

// Initialize model and context...
const struct llama_vocab * vocab = llama_model_get_vocab(model);

// Method 1: Using the high-level generate() API
std::vector<std::string> options = {" Paris", " London", " Berlin"};
generate_params params;
params.max_tokens = 1;
params.custom_sampler = select_sampler(vocab, options);

generate_result result = generate(ctx, vocab, params);
std::cout << "Selected: " << result.text << std::endl;
```

#### 2. `generate()` - Free-form Generation

```cpp
#include "constrained_generation.h"

// Generate with max tokens
generate_params params;
params.max_tokens = 50;
params.temperature = 0.8f;

generate_result result = generate(ctx, vocab, params);
std::cout << result.text << std::endl;

// Generate with stop sequences
params.stop_sequences = {"\nQ:", "\n\n"};
result = generate(ctx, vocab, params);
if (result.stopped_by_sequence) {
    std::cout << "Stopped by: " << result.stop_sequence << std::endl;
}
```

### Low-Level Token Filter API

```cpp
#include "token_filter_sampler.h"

// Manual token filtering
std::vector<llama_token> allowed_tokens = {1234, 5678, 9012};

auto sparams = llama_sampler_chain_default_params();
llama_sampler * smpl = llama_sampler_chain_init(sparams);

llama_sampler_chain_add(smpl, llama_sampler_init_token_filter(allowed_tokens, true));
llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));

llama_token next_token = llama_sampler_sample(smpl, ctx, -1);
```

### Running the Examples

```bash
# Ultra-simple API (recommended)
./build/simple_example models/model.gguf

# Structured thinking with XML tags
./build/thinking_chat_example models/model.gguf

# Mid-level API examples
./build/select_example models/model.gguf
./build/generate_example models/model.gguf

# Pattern constraints
./build/pattern_example models/model.gguf

# Low-level token filtering
./build/example models/model.gguf
```

## API Reference

### `llama_sampler_init_token_filter`

```cpp
struct llama_sampler * llama_sampler_init_token_filter(
    const std::vector<llama_token> & allowed_tokens,
    bool is_allowlist = true
);
```

- `allowed_tokens`: Vector of token IDs to allow/block
- `is_allowlist`: If true, only these tokens allowed; if false, these tokens blocked
- Returns: Initialized sampler ready to add to chain

### `llama_sampler_init_token_filter_set`

```cpp
struct llama_sampler * llama_sampler_init_token_filter_set(
    const std::unordered_set<llama_token> & token_set,
    bool is_allowlist = true
);
```

Same as above but accepts `unordered_set` for efficient lookups with large token sets.

### `llama_sampler_init_stop_sequence`

```cpp
struct llama_sampler * llama_sampler_init_stop_sequence(
    const struct llama_vocab * vocab,
    const std::vector<std::string> & stop_sequences
);
```

Creates a sampler that prevents malformed stop sequences (like XML/JSON tags):

- `vocab`: The model's vocabulary (for tokenization)
- `stop_sequences`: Strings that should properly terminate generation (e.g., `{"</think>", "</output>"}`)
- Returns: Initialized sampler that forces proper tag completion

**How it works:**
1. Tracks accumulated text as tokens are generated
2. When text ends with partial stop sequence (e.g., `</think`), it filters the token distribution
3. Only allows tokens that complete the sequence (e.g., `>`)
4. Blocks tokens that would create malformed patterns (e.g., `</` which would create `</think</`)

**Example use case:**
```cpp
// Prevent malformed XML tags like </think</think</think
generate_params params;
params.stop_sequences = {"</think>"};
params.max_tokens = 300;

// Sampler automatically added in generate() - ensures clean </think> tags!
generate_result result = generate(ctx, vocab, params);
```

### `llama_sampler_init_prefix_select`

```cpp
struct llama_sampler * llama_sampler_init_prefix_select(
    const struct llama_vocab * vocab,
    const std::vector<std::string> & options
);
```

Creates a sampler that forces selection from exact string matches:

- `vocab`: The model's vocabulary
- `options`: List of allowed strings (e.g., `{"<think>", "<output>"}`)
- Returns: Sampler that only allows tokens matching one of the options

Used internally by `LLMSession::select()` for reliable multi-token option selection.

## How It Works

1. Sampler receives token candidates array from previous samplers in chain
2. Iterates through candidates, keeping only allowed tokens (or removing blocked ones)
3. Compacts array in-place, updating size
4. Subsequent samplers in chain operate on filtered set
5. Final sampler (e.g., `dist` or `greedy`) selects from filtered candidates

## API Comparison

Three levels of abstraction for different use cases:

| Feature | Ultra-Simple | Mid-Level | Low-Level |
|---------|-------------|-----------|-----------|
| **Initialization** | `LLMSession llm("model.gguf")` | Manual context setup | Full llama.cpp control |
| **Select** | `llm.select(options)` | `generate()` with custom sampler | `llama_sampler_init_token_filter()` |
| **Generate** | `llm.generate(50, {"\n"})` | `generate(ctx, vocab, params)` | Manual sampler chain |
| **Cleanup** | Automatic (RAII) | Manual | Manual |
| **Use Case** | Rapid prototyping | More control needed | Maximum flexibility |

## License

MIT
