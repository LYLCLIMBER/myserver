cmake_minimum_required(VERSION 3.23)

project(
  myserver
  VERSION 0.1.0
  LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(myserver)

# 避免输出只有 GCC 认识的代码
set_target_properties(myserver
  PROPERTIES
    CXX_EXTENSIONS OFF
)

target_sources(myserver
  PRIVATE
    src/main.cpp
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
