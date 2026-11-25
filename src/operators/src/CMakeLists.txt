cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
pybind11_add_module(${SKBUILD_PROJECT_NAME} binding.cpp)
target_include_directories(${SKBUILD_PROJECT_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(${SKBUILD_PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(${SKBUILD_PROJECT_NAME}
    PRIVATE $<$<CONFIG:Release>:-O3>
)

install(TARGETS ${SKBUILD_PROJECT_NAME} DESTINATION .)