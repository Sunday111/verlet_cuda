find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)
target_link_libraries(verlet_cuda PRIVATE Thrust)
