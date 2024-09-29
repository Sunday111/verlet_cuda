find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)
target_link_libraries(atomic_cas PRIVATE Thrust)
