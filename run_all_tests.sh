#!/bin/bash

# Allow model path to be passed as argument
if [ -n "$1" ]; then
    MODEL_PATH="$1"
else
    MODEL_PATH="models/Qwen2.5-14B.Q4_K_M.gguf"
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Usage: $0 [model_path]"
    exit 1
fi

echo "=========================================="
echo "Running All llama-constrain Tests"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo ""

TESTS=(
    "test_prefix_issue:Prefix issue regression test"
    "simple_example:Basic API usage"
    "select_example:Option selection"
    "generate_example:Free-form generation"
    "pattern_example:Pattern constraints"
    "stopping_example:Stop sequences"
    "multitoken_test:Multi-token handling"
    "thinking_chat_example:Structured thinking"
    "memory_agent_example:3-way agent choices"
)

PASSED=0
FAILED=0
SKIPPED=0

for test in "${TESTS[@]}"; do
    IFS=':' read -r exe desc <<< "$test"

    echo "----------------------------------------"
    echo "Test: $desc ($exe)"
    echo "----------------------------------------"

    if [ ! -f "build/$exe" ]; then
        echo "⊘ SKIP: Executable not found"
        ((SKIPPED++))
        echo ""
        continue
    fi

    # Run and capture output
    if ./build/$exe "$MODEL_PATH" 2>/dev/null > /tmp/test_output_$$.txt; then
        echo "✓ PASS"
        ((PASSED++))
        # Show first few lines of output
        head -10 /tmp/test_output_$$.txt
        echo "..."
    else
        EXIT_CODE=$?
        echo "✗ FAIL: Exit code $EXIT_CODE"
        ((FAILED++))
        # Show last few lines on failure
        tail -20 /tmp/test_output_$$.txt
    fi

    rm -f /tmp/test_output_$$.txt
    echo ""
done

echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Total:   $((PASSED + FAILED + SKIPPED))"
echo "Passed:  $PASSED"
echo "Failed:  $FAILED"
echo "Skipped: $SKIPPED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✓ All tests passed!"
    exit 0
else
    echo "✗ Some tests failed"
    exit 1
fi
