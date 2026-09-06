cmake_minimum_required(VERSION 3.29)

project(verlet_cuda)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
# Set output directories for binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
# Set output directories for archives
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
# Set output directories for libraries
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

# Set path to cloned repositories sources
set(YAE_CLONED_REPOSITORIES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cloned_repositories" CACHE PATH "Path to YAE cloned repository checkouts")

set(YAE_SUPPORT_ROOT "${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/yae-support/main")
set(YAE_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${YAE_SUPPORT_ROOT}/cmake")


include(yae_lto)
enable_lto_globally()
set(YAE_STAGING_ROOT "${CMAKE_BINARY_DIR}/.yae-staging")
set(YAE_STAGING_PLAN "${YAE_STAGING_ROOT}/active-manifests.txt")
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
execute_process(
    COMMAND ${Python3_EXECUTABLE}
            "${YAE_SUPPORT_ROOT}/scripts/stage_directories.py"
            --write-plan
            --manifest-root "${YAE_STAGING_ROOT}"
            --active-manifests "${YAE_STAGING_PLAN}"
            --plan-entry "manifest\tcontent/klvk_compute_shader_example_copy_files.manifest"
            --plan-entry "source\t${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/compute_shader/content"
            --plan-entry "manifest\tcontent/klvk_copy_files.manifest"
            --plan-entry "source\t${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/klvk/content"
            --plan-entry "manifest\tcontent/klvk_curve_fractal_example_copy_files.manifest"
            --plan-entry "source\t${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/curve_fractal/content"
            --plan-entry "manifest\tcontent/klvk_fractal_example_copy_files.manifest"
            --plan-entry "source\t${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/fractal/content"
            --plan-entry "manifest\tcontent/klvk_geometry_shader_example_copy_files.manifest"
            --plan-entry "source\t${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/geometry_shader/content"
            --plan-entry "manifest\tcontent/klvk_minimal_cube_example_copy_files.manifest"
            --plan-entry "source\t${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/minimal_cube/content"
            --plan-entry "manifest\tcontent/klvk_minimal_quad_example_copy_files.manifest"
            --plan-entry "source\t${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/minimal_quad/content"
            --plan-entry "manifest\tcontent/klvk_post_processing_example_copy_files.manifest"
            --plan-entry "source\t${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/post_processing/content"
            --plan-entry "manifest\tcontent/klvk_render_to_texture_example_copy_files.manifest"
            --plan-entry "source\t${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/render_to_texture/content"
            --plan-entry "manifest\tcontent/klvk_simple_lit_cube_example_copy_files.manifest"
            --plan-entry "source\t${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/simple_lit_cube/content"
            --plan-entry "manifest\tcontent/klvk_stencil_example_copy_files.manifest"
            --plan-entry "source\t${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/stencil/content"
            --plan-entry "manifest\tcontent/klvk_text_example_copy_files.manifest"
            --plan-entry "source\t${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/text/content"
            --plan-entry "manifest\tcontent/klvk_textured_quad_example_copy_files.manifest"
            --plan-entry "source\t${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/textured_quad/content"
            --plan-entry "manifest\tcontent/klvk_two_textures_example_copy_files.manifest"
            --plan-entry "source\t${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/two_textures/content"
            --plan-entry "manifest\tcontent/verlet_cuda_copy_files.manifest"
            --plan-entry "source\t${CMAKE_CURRENT_SOURCE_DIR}/src/verlet_cuda/content"
    RESULT_VARIABLE YAE_STAGING_PLAN_RESULT
    ERROR_VARIABLE YAE_STAGING_PLAN_ERROR)
if(NOT YAE_STAGING_PLAN_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to publish staging plan: ${YAE_STAGING_PLAN_ERROR}")
endif()
add_custom_target(yae_reconcile_staging ALL
    COMMAND ${Python3_EXECUTABLE} "${YAE_SUPPORT_ROOT}/scripts/stage_directories.py"
            --reconcile
            --destination-root "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
            --manifest-root "${YAE_STAGING_ROOT}"
            --active-manifests "${YAE_STAGING_PLAN}"
            --legacy-manifest-root "${CMAKE_BINARY_DIR}/yae_modules"
    COMMENT "reconcile staged content"
    VERBATIM)


# https://github.com/fmtlib/fmt 12.2.0
set(YAE_fmtlib_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/fmtlib/fmt/12.2.0)
add_subdirectory(${YAE_fmtlib_SOURCES} yae_modules/fmtlib/fmt/12.2.0 SYSTEM)

set(YAE_edt_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/edt/main/modules/edt)
add_subdirectory(${YAE_edt_SOURCES} yae_modules/Sunday111/edt/main/modules/edt SYSTEM)

set(YAE_ass_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/ass/main/modules/ass)
add_subdirectory(${YAE_ass_SOURCES} yae_modules/Sunday111/ass/main/modules/ass SYSTEM)

set(YAE_refl_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/refl/main/modules/refl)
add_subdirectory(${YAE_refl_SOURCES} yae_modules/Sunday111/refl/main/modules/refl SYSTEM)

# https://github.com/madler/zlib v1.3.2
set(YAE_zlib_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/madler/zlib/v1.3.2)
option(ZLIB_BUILD_SHARED "" OFF)
option(ZLIB_BUILD_STATIC "" ON)
option(ZLIB_BUILD_TESTING "" OFF)
option(ZLIB_INSTALL "" OFF)
add_subdirectory(${YAE_zlib_SOURCES} yae_modules/madler/zlib/v1.3.2 SYSTEM)

include(${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/dependencies/zlib_target.cmake)
# https://github.com/facebook/zstd v1.5.7
set(YAE_zstd_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/facebook/zstd/v1.5.7)
option(ZSTD_BUILD_PROGRAMS "" OFF)
option(ZSTD_BUILD_CONTRIB "" OFF)
option(ZSTD_BUILD_TESTS "" OFF)
option(ZSTD_BUILD_STATIC "" ON)
option(ZSTD_BUILD_SHARED "" OFF)
option(ZSTD_LEGACY_SUPPORT "" OFF)
add_subdirectory(${YAE_zstd_SOURCES}/build/cmake yae_modules/facebook/zstd/v1.5.7 SYSTEM)

# https://github.com/davea42/libdwarf-code v2.3.2
set(YAE_libdwarf_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/davea42/libdwarf-code/v2.3.2)
option(BUILD_NON_SHARED "" ON)
option(BUILD_SHARED "" OFF)
option(BUILD_DWARFDUMP "" OFF)
option(BUILD_DWARFGEN "" OFF)
option(BUILD_DWARFEXAMPLE "" OFF)
option(DO_TESTING "" OFF)
option(PIC_ALWAYS "" ON)
add_subdirectory(${YAE_libdwarf_SOURCES} yae_modules/davea42/libdwarf-code/v2.3.2 SYSTEM)

include(${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/dependencies/Findlibdwarf.cmake)
# https://github.com/jeremy-rifkin/cpptrace main
set(YAE_cpptrace_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/jeremy-rifkin/cpptrace/main)
option(CPPTRACE_USE_EXTERNAL_LIBDWARF "" ON)
add_subdirectory(${YAE_cpptrace_SOURCES} yae_modules/jeremy-rifkin/cpptrace/main SYSTEM)

# https://github.com/Neargye/magic_enum v0.9.8
set(YAE_magic_enum_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Neargye/magic_enum/v0.9.8)
add_subdirectory(${YAE_magic_enum_SOURCES} yae_modules/Neargye/magic_enum/v0.9.8 SYSTEM)

# https://github.com/martinus/unordered_dense v4.9.2
set(YAE_unordered_dense_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/martinus/unordered_dense/v4.9.2)
add_subdirectory(${YAE_unordered_dense_SOURCES} yae_modules/martinus/unordered_dense/v4.9.2 SYSTEM)

# https://github.com/nlohmann/json develop
set(YAE_nlohmann_json_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/nlohmann/json/develop)
add_subdirectory(${YAE_nlohmann_json_SOURCES} yae_modules/nlohmann/json/develop SYSTEM)

# https://github.com/KhronosGroup/Vulkan-Headers v1.4.360
set(YAE_vulkan_headers_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/KhronosGroup/Vulkan-Headers/v1.4.360)
add_subdirectory(${YAE_vulkan_headers_SOURCES} yae_modules/KhronosGroup/Vulkan-Headers/v1.4.360 SYSTEM)

# https://github.com/Sunday111/Vulkan-Hpp cpptrace
set(YAE_vulkan_hpp_sunday_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/Vulkan-Hpp/cpptrace)
option(VULKAN_HPP_BUILD_CXX_MODULE "" OFF)
option(VULKAN_HPP_BUILD_WITH_LOCAL_VULKAN_HPP "" ON)
option(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC "" ON)
option(VULKAN_HPP_GENERATOR_BUILD "" OFF)
option(VULKAN_HPP_PRECOMPILE "" OFF)
option(VULKAN_HPP_SAMPLES_BUILD "" OFF)
option(VULKAN_HPP_TESTS_BUILD "" OFF)
option(VULKAN_HPP_USE_CPPTRACE "" ON)
add_subdirectory(${YAE_vulkan_hpp_sunday_SOURCES} yae_modules/Sunday111/Vulkan-Hpp/cpptrace SYSTEM)

# https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator v3.4.0
set(YAE_vma_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/v3.4.0)
add_subdirectory(${YAE_vma_SOURCES} yae_modules/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/v3.4.0 SYSTEM)

# https://github.com/ocornut/imgui v1.92.9
set(YAE_imgui_sources_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/ocornut/imgui/v1.92.9)
# https://github.com/glfw/glfw master
set(YAE_glfw_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/glfw/glfw/master)
option(GLFW_BUILD_EXAMPLES "" OFF)
option(GLFW_BUILD_TESTS "" OFF)
option(GLFW_BUILD_DOCS "" OFF)
add_subdirectory(${YAE_glfw_SOURCES} yae_modules/glfw/glfw/master SYSTEM)

# https://github.com/freetype/freetype VER-2-14-3
set(YAE_freetype_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/freetype/freetype/VER-2-14-3)
option(FT_DISABLE_HARFBUZZ "" ON)
option(FT_DISABLE_BROTLI "" ON)
option(FT_DISABLE_PNG "" ON)
option(FT_DISABLE_BZIP2 "" ON)
option(FT_DISABLE_ZLIB "" ON)
add_subdirectory(${YAE_freetype_SOURCES} yae_modules/freetype/freetype/VER-2-14-3 SYSTEM)

set(YAE_imgui_vulkan_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/imgui_vulkan)
add_subdirectory(${YAE_imgui_vulkan_SOURCES} yae_modules/Sunday111/klvk/main/imgui_vulkan SYSTEM)

# https://github.com/gabime/spdlog v1.17.0
set(YAE_spdlog_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/gabime/spdlog/v1.17.0)
option(SPDLOG_BUILD_EXAMPLE "" OFF)
option(SPDLOG_BUILD_TESTS "" OFF)
option(SPDLOG_FMT_EXTERNAL "" ON)
add_subdirectory(${YAE_spdlog_SOURCES} yae_modules/gabime/spdlog/v1.17.0 SYSTEM)

# binary dependency slang (linux-x86_64)
find_package(slang CONFIG REQUIRED PATHS ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/yae-slang/v2026.14/.prebuilt/linux-x86_64 NO_DEFAULT_PATH)

# https://github.com/nothings/stb master
set(YAE_stb_sources_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/nothings/stb/master)
set(YAE_stb_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/stb)
add_subdirectory(${YAE_stb_SOURCES} yae_modules/Sunday111/klvk/main/stb SYSTEM)

set(YAE_klvk_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/klvk)
add_subdirectory(${YAE_klvk_SOURCES} yae_modules/Sunday111/klvk/main/klvk SYSTEM)

include(${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/klvk/ffmpeg.cmake)
set(YAE_verlet_cuda_SOURCES src/verlet_cuda)
add_subdirectory(${YAE_verlet_cuda_SOURCES} yae_modules/src/verlet_cuda SYSTEM)

include(${CMAKE_CURRENT_SOURCE_DIR}/src/verlet_cuda/add_cuda.cmake)
# https://github.com/google/benchmark v1.9.5
set(YAE_gbench_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/google/benchmark/v1.9.5)
option(BENCHMARK_ENABLE_TESTING "" OFF)
add_subdirectory(${YAE_gbench_SOURCES} yae_modules/google/benchmark/v1.9.5 SYSTEM)

# https://github.com/google/googletest v1.18.0
set(YAE_gtest_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/google/googletest/v1.18.0)
option(INSTALL_GTEST "" OFF)
option(BUILD_GMOCK "" OFF)
add_subdirectory(${YAE_gtest_SOURCES} yae_modules/google/googletest/v1.18.0 SYSTEM)

set(YAE_yae_example_benchmark_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/yae-support/main/modules/examples/yae_example_benchmark)
add_subdirectory(${YAE_yae_example_benchmark_SOURCES} yae_modules/Sunday111/yae-support/main/modules/examples/yae_example_benchmark SYSTEM)

set(YAE_yae_example_benchmark_main_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/yae-support/main/modules/examples/yae_example_benchmark_main)
add_subdirectory(${YAE_yae_example_benchmark_main_SOURCES} yae_modules/Sunday111/yae-support/main/modules/examples/yae_example_benchmark_main SYSTEM)

set(YAE_yae_example_library_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/yae-support/main/modules/examples/yae_example_library)
add_subdirectory(${YAE_yae_example_library_SOURCES} yae_modules/Sunday111/yae-support/main/modules/examples/yae_example_library SYSTEM)

set(YAE_yae_example_header_only_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/yae-support/main/modules/examples/yae_example_header_only)
add_subdirectory(${YAE_yae_example_header_only_SOURCES} yae_modules/Sunday111/yae-support/main/modules/examples/yae_example_header_only SYSTEM)

set(YAE_yae_example_executable_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/yae-support/main/modules/examples/yae_example_executable)
add_subdirectory(${YAE_yae_example_executable_SOURCES} yae_modules/Sunday111/yae-support/main/modules/examples/yae_example_executable SYSTEM)

set(YAE_yae_example_tests_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/yae-support/main/modules/examples/yae_example_tests)
add_subdirectory(${YAE_yae_example_tests_SOURCES} yae_modules/Sunday111/yae-support/main/modules/examples/yae_example_tests SYSTEM)

set(YAE_yae_example_tests_main_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/yae-support/main/modules/examples/yae_example_tests_main)
add_subdirectory(${YAE_yae_example_tests_main_SOURCES} yae_modules/Sunday111/yae-support/main/modules/examples/yae_example_tests_main SYSTEM)

set(YAE_klvk_application_frame_pacing_test_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/tests/application_frame_pacing)
add_subdirectory(${YAE_klvk_application_frame_pacing_test_SOURCES} yae_modules/Sunday111/klvk/main/tests/application_frame_pacing SYSTEM)

set(YAE_klvk_compute_shader_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/compute_shader)
add_subdirectory(${YAE_klvk_compute_shader_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/compute_shader SYSTEM)

set(YAE_klvk_curve_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/curve)
add_subdirectory(${YAE_klvk_curve_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/curve SYSTEM)

set(YAE_klvk_curve_fractal_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/curve_fractal)
add_subdirectory(${YAE_klvk_curve_fractal_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/curve_fractal SYSTEM)

set(YAE_klvk_diagnostic_config_test_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/tests/diagnostic_config)
add_subdirectory(${YAE_klvk_diagnostic_config_test_SOURCES} yae_modules/Sunday111/klvk/main/tests/diagnostic_config SYSTEM)

set(YAE_klvk_diagnostic_replay_test_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/tests/diagnostic_replay)
add_subdirectory(${YAE_klvk_diagnostic_replay_test_SOURCES} yae_modules/Sunday111/klvk/main/tests/diagnostic_replay SYSTEM)

include(${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/tests/diagnostic_replay/ffmpeg.cmake)
include(${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/tests/diagnostic_replay/private_headers.cmake)
set(YAE_klvk_event_manager_test_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/tests/event_manager)
add_subdirectory(${YAE_klvk_event_manager_test_SOURCES} yae_modules/Sunday111/klvk/main/tests/event_manager SYSTEM)

set(YAE_klvk_falling_sand_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/falling_sand)
add_subdirectory(${YAE_klvk_falling_sand_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/falling_sand SYSTEM)

set(YAE_klvk_fractal_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/fractal)
add_subdirectory(${YAE_klvk_fractal_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/fractal SYSTEM)

set(YAE_klvk_geometry_shader_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/geometry_shader)
add_subdirectory(${YAE_klvk_geometry_shader_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/geometry_shader SYSTEM)

set(YAE_klvk_minimal_cube_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/minimal_cube)
add_subdirectory(${YAE_klvk_minimal_cube_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/minimal_cube SYSTEM)

set(YAE_klvk_minimal_quad_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/minimal_quad)
add_subdirectory(${YAE_klvk_minimal_quad_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/minimal_quad SYSTEM)

set(YAE_klvk_painter2d_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/painter2d)
add_subdirectory(${YAE_klvk_painter2d_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/painter2d SYSTEM)

set(YAE_klvk_pendulum_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/pendulum)
add_subdirectory(${YAE_klvk_pendulum_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/pendulum SYSTEM)

set(YAE_klvk_perf_recorder_test_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/tests/perf_recorder)
add_subdirectory(${YAE_klvk_perf_recorder_test_SOURCES} yae_modules/Sunday111/klvk/main/tests/perf_recorder SYSTEM)

set(YAE_klvk_post_processing_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/post_processing)
add_subdirectory(${YAE_klvk_post_processing_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/post_processing SYSTEM)

set(YAE_klvk_render_to_texture_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/render_to_texture)
add_subdirectory(${YAE_klvk_render_to_texture_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/render_to_texture SYSTEM)

set(YAE_klvk_shader_cache_test_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/tests/shader_cache)
add_subdirectory(${YAE_klvk_shader_cache_test_SOURCES} yae_modules/Sunday111/klvk/main/tests/shader_cache SYSTEM)

include(${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/tests/shader_cache/private_headers.cmake)
set(YAE_klvk_simple_lit_cube_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/simple_lit_cube)
add_subdirectory(${YAE_klvk_simple_lit_cube_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/simple_lit_cube SYSTEM)

set(YAE_klvk_stencil_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/stencil)
add_subdirectory(${YAE_klvk_stencil_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/stencil SYSTEM)

set(YAE_klvk_tetris_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/tetris)
add_subdirectory(${YAE_klvk_tetris_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/tetris SYSTEM)

set(YAE_klvk_text_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/text)
add_subdirectory(${YAE_klvk_text_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/text SYSTEM)

set(YAE_klvk_textured_quad_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/textured_quad)
add_subdirectory(${YAE_klvk_textured_quad_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/textured_quad SYSTEM)

set(YAE_klvk_timer_manager_test_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/tests/timer_manager)
add_subdirectory(${YAE_klvk_timer_manager_test_SOURCES} yae_modules/Sunday111/klvk/main/tests/timer_manager SYSTEM)

set(YAE_klvk_two_textures_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/two_textures)
add_subdirectory(${YAE_klvk_two_textures_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/two_textures SYSTEM)

set(YAE_klvk_vulkan_hpp_test_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/tests/vulkan_hpp)
add_subdirectory(${YAE_klvk_vulkan_hpp_test_SOURCES} yae_modules/Sunday111/klvk/main/tests/vulkan_hpp SYSTEM)

set(YAE_imgui_glfw_cmake_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/imgui_glfw_cmake/yae)
add_subdirectory(${YAE_imgui_glfw_cmake_SOURCES} yae_modules/Sunday111/imgui_glfw_cmake/yae SYSTEM)

set(YAE_refl_tests_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/refl/main/modules/refl_tests)
add_subdirectory(${YAE_refl_tests_SOURCES} yae_modules/Sunday111/refl/main/modules/refl_tests SYSTEM)

set(YAE_edt_tests_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/edt/main/modules/edt_tests)
add_subdirectory(${YAE_edt_tests_SOURCES} yae_modules/Sunday111/edt/main/modules/edt_tests SYSTEM)

set(YAE_AssTests_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/ass/main/modules/AssTests)
add_subdirectory(${YAE_AssTests_SOURCES} yae_modules/Sunday111/ass/main/modules/AssTests SYSTEM)

if(TARGET yae_reconcile_staging AND TARGET edt)
    add_dependencies(edt yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET ass)
    add_dependencies(ass yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET refl)
    add_dependencies(refl yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET imgui_vulkan)
    add_dependencies(imgui_vulkan yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET stb)
    add_dependencies(stb yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk)
    add_dependencies(klvk yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET verlet_cuda)
    add_dependencies(verlet_cuda yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET yae_example_benchmark)
    add_dependencies(yae_example_benchmark yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET yae_example_benchmark_main)
    add_dependencies(yae_example_benchmark_main yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET yae_example_library)
    add_dependencies(yae_example_library yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET yae_example_header_only)
    add_dependencies(yae_example_header_only yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET yae_example_executable)
    add_dependencies(yae_example_executable yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET yae_example_tests)
    add_dependencies(yae_example_tests yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET yae_example_tests_main)
    add_dependencies(yae_example_tests_main yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_application_frame_pacing_test)
    add_dependencies(klvk_application_frame_pacing_test yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_compute_shader_example)
    add_dependencies(klvk_compute_shader_example yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_curve_example)
    add_dependencies(klvk_curve_example yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_curve_fractal_example)
    add_dependencies(klvk_curve_fractal_example yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_diagnostic_config_test)
    add_dependencies(klvk_diagnostic_config_test yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_diagnostic_replay_test)
    add_dependencies(klvk_diagnostic_replay_test yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_event_manager_test)
    add_dependencies(klvk_event_manager_test yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_falling_sand_example)
    add_dependencies(klvk_falling_sand_example yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_fractal_example)
    add_dependencies(klvk_fractal_example yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_geometry_shader_example)
    add_dependencies(klvk_geometry_shader_example yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_minimal_cube_example)
    add_dependencies(klvk_minimal_cube_example yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_minimal_quad_example)
    add_dependencies(klvk_minimal_quad_example yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_painter2d_example)
    add_dependencies(klvk_painter2d_example yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_pendulum_example)
    add_dependencies(klvk_pendulum_example yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_perf_recorder_test)
    add_dependencies(klvk_perf_recorder_test yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_post_processing_example)
    add_dependencies(klvk_post_processing_example yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_render_to_texture_example)
    add_dependencies(klvk_render_to_texture_example yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_shader_cache_test)
    add_dependencies(klvk_shader_cache_test yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_simple_lit_cube_example)
    add_dependencies(klvk_simple_lit_cube_example yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_stencil_example)
    add_dependencies(klvk_stencil_example yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_tetris_example)
    add_dependencies(klvk_tetris_example yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_text_example)
    add_dependencies(klvk_text_example yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_textured_quad_example)
    add_dependencies(klvk_textured_quad_example yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_timer_manager_test)
    add_dependencies(klvk_timer_manager_test yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_two_textures_example)
    add_dependencies(klvk_two_textures_example yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET klvk_vulkan_hpp_test)
    add_dependencies(klvk_vulkan_hpp_test yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET imgui_glfw_cmake)
    add_dependencies(imgui_glfw_cmake yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET refl_tests)
    add_dependencies(refl_tests yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET edt_tests)
    add_dependencies(edt_tests yae_reconcile_staging)
endif()
if(TARGET yae_reconcile_staging AND TARGET AssTests)
    add_dependencies(AssTests yae_reconcile_staging)
endif()

enable_testing()
