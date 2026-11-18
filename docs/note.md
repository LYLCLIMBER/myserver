## 2026-11-15

sample.cpp 实测，使用 `qwen2.5-3b-instrcut-q4_k_m.gguf`

batch_size = 1，权重 2 GB，参数量 3B，load_time 加载模型到内存 / 显存的速度

- `-ngl 99`，纯 GPU 93 t/s，load_time = 1014 ms
- `-ngl 0`，纯 CPU 23 t/s，load_time = 934 ms

稠密模型的前向浮点数计算次数估算：2P

4060 显存带宽 272 GB/s，CPU 读内存带宽 50 GB/s，利用率分别为 68% / 90%
4060 笔记本端浮点数算力为 14.56 TFLOPS，GPU 计算利用率 4%

结论：
- GPU 端 decode 的速度主要受限于显存带宽
- GPU 更快的原因是显存带宽更大，大量的计算资源目前是闲置的

## 2026-11-17

安装并导入了 llama.cpp

推荐的以交互式查看项目配置的方式，但不建议用它配置项目，因为难以记录历史

```shell
ccmake build-cuda
```

find_package 默认去找 `<prefix>/lib/cmake/<name>`
- `<prefix>`：可以用 `CMAKE_PREFIX_PATH` 设置
- `<name>`：`{Package_name}Config.cmake` 或 `{Package_name}-config.cmake`

使用 llama 包不需要 ggml 包，因为它自己已经导入了
