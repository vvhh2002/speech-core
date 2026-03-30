cmake_minimum_required(VERSION 3.16)
project(speech_core VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(FetchContent)

# 1. 声明 SIMDe 的下载信息
FetchContent_Declare(
  simde
  GIT_REPOSITORY         https://github.com/simd-everywhere/simde.git
  GIT_TAG                master
  GIT_PROGRESS           TRUE
  USES_TERMINAL_DOWNLOAD TRUE
  GIT_CONFIG             "http.proxy=http://127.0.0.1:1087"
  GIT_SHALLOW            TRUE  # 推荐添加，减少克隆时间
)

# 2. 下载并配置
FetchContent_MakeAvailable(simde)

# Library sources
set(SOURCES
    src/pipeline/voice_pipeline.cpp
    src/pipeline/turn_detector.cpp
    src/pipeline/speech_queue.cpp
    src/pipeline/conversation_context.cpp
    src/vad/streaming_vad.cpp
    src/audio/audio_buffer.cpp
    src/audio/pcm_codec.cpp
    src/tools/tool_registry.cpp
    src/tools/intent_matcher.cpp
    src/tools/tool_executor.cpp
    src/speech_core_c.cpp
)

option(SPEECH_CORE_ENABLE_SIMDE "Enable SIMDE optimized resampler" ON)

if(SPEECH_CORE_ENABLE_SIMDE)
    list(APPEND SOURCES src/audio/resampler_simde.cpp)
else()
    list(APPEND SOURCES src/audio/resampler.cpp)
endif()

add_library(speech_core ${SOURCES})

# --- 关键修改点 ---

# 3. 链接 simde 并处理包含路径
# 使用 INTERFACE 确保安装后下游也能找到 simde
target_link_libraries(speech_core PRIVATE simde)

target_include_directories(speech_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        # 如果你的头文件里 #include 了 simde，这里需要暴露 simde 的路径
        $<BUILD_INTERFACE:${simde_SOURCE_DIR}> 
        $<INSTALL_INTERFACE:include>
)

# 4. 编译选项
target_compile_options(speech_core PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O2>
)

# 5. 安装部分调整
install(TARGETS speech_core
    EXPORT speech_core-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# 安装你自己的头文件
install(DIRECTORY include/
    DESTINATION include
)

# 如果你希望用户安装你的 speech_core 时，本地也有一份 simde 副本：
install(DIRECTORY "${simde_SOURCE_DIR}/simde"
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# 导出 target 信息 (方便下游 find_package)
install(EXPORT speech_core-targets
    FILE speech_coreTargets.cmake
    NAMESPACE speech_core::
    DESTINATION lib/cmake/speech_core
)

# Tests
option(SPEECH_CORE_BUILD_TESTS "Build tests" ON)

if(SPEECH_CORE_BUILD_TESTS)
    enable_testing()

    file(GLOB TEST_SOURCES tests/test_*.cpp)
    foreach(TEST_SRC ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
        add_executable(${TEST_NAME} ${TEST_SRC})
        target_link_libraries(${TEST_NAME} PRIVATE speech_core)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()
endif()
