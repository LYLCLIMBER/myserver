# roadmap

## 整体路线图

### M0 构建工程骨架 （状态：完成）

**目标**：让项目使用 CMake 构建代码，并将项目纳入 Git 管理

**任务**
- [x] 编写 CmakeLists.txt，构建一个 hello world
- [x] 创建 git 仓库，编写 .gitignore 并同步到远程仓库
- [x] 添加许可证

**完成标准**
- 项目能够受 Cmake 和 git 管理

### M1 命令行推理（状态：未完成）

**目标**：实现用命令行加载模型进行推理

**任务**
- [x] 阅读 llama.cpp/examples/simple/simple.cpp，画出流程图
- [x] 下载 Qwen2.5-3B-Instruct 的 Q4_K_M 量化版，编译 llama.cpp 跑通一次
- [x] 让 myserver 链接上 llama.cpp
- [ ] 按照 simple.cpp 流程编写 main.cpp，完成推理
  - [x] s1：加载模型，然后释放
  - [x] s2：分词
  - [x] s3：创建 context 和 sample
  - [ ] s4：处理 prompt，采样出一个 token
  - [ ] s5：循环生成，直到遇到结束符或达到 n_predict
- [ ] 加上命令行参数

**完成标准**
- s1：打印模型的一项信息然后正常退出
- s2：打印 token id，再用 llama_token_to_piece 把它们拼回去，结果和原来的 prompt 一致
- s3：创建成功，释放顺序正确，没有崩溃
- s4：打印出一个合理的下一个词
- s5：能够输出一整句话
- 编译零警告

- [ ] M2 代码分层

**任务**
- [ ] 加入 clang-tidy

将推理逻辑拆分为独立的模块，做好接口设计和职责划分

- [ ] M3 最小 HTTP 服务

POST /completion 能够返回推理结果，GET /health 返回 OK

- [ ] M4 兼容 OpenAI 接口

能连接到现有的聊天客户端上

- [ ] M5 流式输出

能够像 ChatGPT 那样吐字输出

- [ ] M6 产品化

可能加入并发请求、配置、日志、测试等内容。

- [ ] M7 深入

自定义采样、调度和缓存等内容
