cmake_minimum_required(VERSION 3.23)

project(
  myserver
  VERSION 0.1.0
  LANGUAGES CXX
)

add_executable(myserver)

target_sources(myserver
  PRIVATE
    src/main.cpp
)

target_compile_features(myserver
  PUBLIC
    cxx_std_20
)

target_compile_options(myserver
  PRIVATE # 警告设置不传播给其他 Target
    -Wall
    -Wextra
    -Wpedantic
)
