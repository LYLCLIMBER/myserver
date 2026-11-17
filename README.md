# myserver

## 简介

（开发中）一个简单的推理服务器。调用 llama.cpp 对模型进行推理。

## 构建方式

```shell
cmake -B build -S . -DCMAKE_PREFIX_PATH={LLAMA_INSTALL_PATH}
cmake --build build
```

## llama.cpp 配置和安装

llama.cpp 配置指令

```shell
cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_TESTS=OFF
```

- `-DGGML_NATIVE=ON` 针对这台机器的 CPU 开启所有指令集优化，但可能无法在别的设备上运行

llama.cpp 编译指令

```shell
cmake --build build-cuda -j 16
```

llama.cpp 安装指令

```shell
cmake --install build-cuda --prefix ~/Codes/project/llama-install
```
