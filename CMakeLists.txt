cmake_minimum_required(VERSION 3.23)

project(
  myserver
  VERSION 0.1.0
  LANGUAGES CXX
)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(llama REQUIRED)

add_executable(myserver)

target_sources(myserver
  PRIVATE
    src/main.cpp
)

target_link_libraries(myserver
  PRIVATE
    llama
)

target_compile_features(myserver
  PRIVATE
    cxx_std_20
)

target_compile_options(myserver
  PRIVATE # 警告设置不传播给其他 Target
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wsign-conversion
)

# 避免输出只有 GCC 认识的代码
set_target_properties(myserver
  PROPERTIES
    CXX_EXTENSIONS OFF
)
