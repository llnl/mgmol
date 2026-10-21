include_directories( ${CMAKE_SOURCE_DIR}/src )

add_executable(check_input check_input.cc)

add_executable(example1 example1.cc)

add_executable(computeDipole computeDipole.cc)

target_include_directories(check_input PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(example1 PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(computeDipole PRIVATE ${Boost_INCLUDE_DIRS})

target_link_libraries(check_input mgmol_src)
target_link_libraries(example1 mgmol_src)
target_link_libraries(computeDipole mgmol_src)
