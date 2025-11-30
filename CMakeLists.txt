cmake_minimum_required(VERSION 3.12)
project(custom_sampler)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp" CACHE PATH "Path to llama.cpp directory")

if(NOT EXISTS "${LLAMA_CPP_DIR}/include/llama.h")
    message(FATAL_ERROR "llama.h not found in ${LLAMA_CPP_DIR}/include/. Did you initialize git submodules?")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${LLAMA_CPP_DIR}/include)
include_directories(${LLAMA_CPP_DIR}/ggml/include)

link_directories(${LLAMA_CPP_DIR}/build/bin)
link_directories(${LLAMA_CPP_DIR}/build/ggml/src)

add_library(token_filter_sampler STATIC
    src/token_filter_sampler.cpp
    include/token_filter_sampler.h
)

add_library(constrained_generation STATIC
    src/constrained_generation.cpp
    include/constrained_generation.h
)

add_library(constrained_llm STATIC
    src/constrained_llm.cpp
    include/constrained_llm.h
)

target_include_directories(token_filter_sampler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(constrained_generation PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(constrained_llm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(constrained_generation
    token_filter_sampler
)

target_link_libraries(constrained_llm
    constrained_generation
    token_filter_sampler
    llama
    ggml
)

add_executable(example examples/example.cpp)
target_link_libraries(example
    token_filter_sampler
    llama
    ggml
    Threads::Threads
)

add_executable(select_example examples/select_example.cpp)
target_link_libraries(select_example
    token_filter_sampler
    llama
    ggml
    Threads::Threads
)

add_executable(generate_example examples/generate_example.cpp)
target_link_libraries(generate_example
    constrained_generation
    token_filter_sampler
    llama
    ggml
    Threads::Threads
)

add_executable(simple_example examples/simple_example.cpp)
target_link_libraries(simple_example
    constrained_llm
    Threads::Threads
)

add_executable(multitoken_test examples/multitoken_test.cpp)
target_link_libraries(multitoken_test
    constrained_llm
    Threads::Threads
)

add_executable(variables_example examples/variables_example.cpp)
target_link_libraries(variables_example
    constrained_llm
    Threads::Threads
)

add_executable(pattern_example examples/pattern_example.cpp)
target_link_libraries(pattern_example
    constrained_llm
    Threads::Threads
)

add_executable(stopping_example examples/stopping_example.cpp)
target_link_libraries(stopping_example
    constrained_llm
    Threads::Threads
)

add_executable(min_max_tokens_example examples/min_max_tokens_example.cpp)
target_link_libraries(min_max_tokens_example
    constrained_llm
    Threads::Threads
)

add_executable(xml_stop_example examples/xml_stop_example.cpp)
target_link_libraries(xml_stop_example
    constrained_llm
    Threads::Threads
)

add_executable(context_save_load_example examples/context_save_load_example.cpp)
target_link_libraries(context_save_load_example
    constrained_llm
    Threads::Threads
)

add_executable(context_speed_test examples/context_speed_test.cpp)
target_link_libraries(context_speed_test
    constrained_llm
    Threads::Threads
)

add_executable(context_memory_example examples/context_memory_example.cpp)
target_link_libraries(context_memory_example
    constrained_llm
    Threads::Threads
)

add_executable(auto_cache_example examples/auto_cache_example.cpp)
target_link_libraries(auto_cache_example
    constrained_llm
    Threads::Threads
)

add_executable(cache_size_test examples/cache_size_test.cpp)
target_link_libraries(cache_size_test
    constrained_llm
    Threads::Threads
)

add_executable(thinking_chat_example examples/thinking_chat_example.cpp)
target_link_libraries(thinking_chat_example
    constrained_llm
    Threads::Threads
)

add_executable(stop_sequence_test examples/stop_sequence_test.cpp)
target_link_libraries(stop_sequence_test
    constrained_llm
    Threads::Threads
)

add_executable(token_debug_test examples/token_debug_test.cpp)
target_link_libraries(token_debug_test
    constrained_llm
    Threads::Threads
)

add_executable(memory_agent_example examples/memory_agent_example.cpp)
target_link_libraries(memory_agent_example
    constrained_llm
    Threads::Threads
)

add_executable(test_prefix_issue examples/test_prefix_issue.cpp)
target_link_libraries(test_prefix_issue
    constrained_llm
    Threads::Threads
)

if(APPLE)
    target_link_libraries(example "-framework Accelerate")
    target_link_libraries(select_example "-framework Accelerate")
    target_link_libraries(generate_example "-framework Accelerate")
    target_link_libraries(simple_example "-framework Accelerate")
    target_link_libraries(multitoken_test "-framework Accelerate")
    target_link_libraries(variables_example "-framework Accelerate")
    target_link_libraries(pattern_example "-framework Accelerate")
    target_link_libraries(stopping_example "-framework Accelerate")
    target_link_libraries(min_max_tokens_example "-framework Accelerate")
    target_link_libraries(xml_stop_example "-framework Accelerate")
    target_link_libraries(context_save_load_example "-framework Accelerate")
    target_link_libraries(context_speed_test "-framework Accelerate")
    target_link_libraries(context_memory_example "-framework Accelerate")
    target_link_libraries(auto_cache_example "-framework Accelerate")
    target_link_libraries(cache_size_test "-framework Accelerate")
    target_link_libraries(thinking_chat_example "-framework Accelerate")
    target_link_libraries(stop_sequence_test "-framework Accelerate")
    target_link_libraries(token_debug_test "-framework Accelerate")
    target_link_libraries(memory_agent_example "-framework Accelerate")
    target_link_libraries(test_prefix_issue "-framework Accelerate")
endif()
