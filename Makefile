.PHONY: all build clean test example run help llama

LLAMA_CPP_DIR := external/llama.cpp
BUILD_DIR := build
MODEL_PATH ?= models/model.gguf

help:
	@echo "Available targets:"
	@echo "  make llama       - Build llama.cpp (required first time)"
	@echo "  make build       - Build the custom sampler library and example"
	@echo "  make example     - Build only the example executable"
	@echo "  make run         - Run the example (set MODEL_PATH=/path/to/model.gguf)"
	@echo "  make test        - Run tests (TODO)"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make clean-all   - Clean everything including llama.cpp build"
	@echo "  make help        - Show this help message"

llama:
	@echo "Building llama.cpp..."
	cd $(LLAMA_CPP_DIR) && cmake -B build && cmake --build build -j4

build: $(BUILD_DIR)/Makefile
	@echo "Building custom sampler..."
	cmake --build $(BUILD_DIR)

$(BUILD_DIR)/Makefile:
	@echo "Configuring build..."
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
	@if [ ! -f compile_commands.json ]; then \
		ln -s $(BUILD_DIR)/compile_commands.json . ; \
	fi

example: build

run: build
	@if [ ! -f "$(MODEL_PATH)" ]; then \
		echo "Error: Model not found at $(MODEL_PATH)"; \
		echo "Usage: make run MODEL_PATH=/path/to/model.gguf"; \
		exit 1; \
	fi
	$(BUILD_DIR)/example $(MODEL_PATH)

test:
	@echo "Tests not yet implemented"

clean:
	rm -rf $(BUILD_DIR)
	rm -f compile_commands.json

clean-all: clean
	rm -rf $(LLAMA_CPP_DIR)/build

all: llama build
