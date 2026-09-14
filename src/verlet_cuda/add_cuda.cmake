find_package(CUDAToolkit REQUIRED)
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)
set_property(TARGET verlet_cuda PROPERTY CUDA_RUNTIME_LIBRARY Shared)
if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^[0-9]+$")
    message(FATAL_ERROR "verlet_cuda requires exactly one numeric CUDA architecture")
endif()
target_compile_options(
    verlet_cuda
    PRIVATE
        $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:--cuda-include-ptx=sm_${CMAKE_CUDA_ARCHITECTURES}>)
target_link_libraries(verlet_cuda PRIVATE Thrust CUDA::toolkit)

set(VERLET_WORLD_WIDTH 1920 CACHE STRING "Physics world width")
set(VERLET_WORLD_HEIGHT 1040 CACHE STRING "Physics world height")
set(VERLET_MAX_OBJECTS 2000000 CACHE STRING "Maximum particle population")
foreach(setting VERLET_WORLD_WIDTH VERLET_WORLD_HEIGHT VERLET_MAX_OBJECTS)
    if(NOT "${${setting}}" MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "${setting} must be a positive integer")
    endif()
endforeach()
target_compile_definitions(
    verlet_cuda PRIVATE
    VERLET_WORLD_WIDTH=${VERLET_WORLD_WIDTH}
    VERLET_WORLD_HEIGHT=${VERLET_WORLD_HEIGHT}
    VERLET_MAX_OBJECTS=${VERLET_MAX_OBJECTS})
