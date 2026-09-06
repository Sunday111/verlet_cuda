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
