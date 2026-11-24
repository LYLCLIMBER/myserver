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

## 2026-11-19

遇到复杂逻辑，可以先想清楚，写好注释再开始写。

sizeof() - 计算对象本身的字节占用，多用来计算 C 风格数组的字节/元素个数；std::string 等 C++ 类型多用 .size()

```C++
std::string s;
s.resize(256);
char buf[256];

std::cout << sizeof(s); // 32, std::string 管理类本身只有 32 个字节，它的字符数据在堆上
std::cout << sizeof(buf);  // 256，buf  指向的就是第一个字符数据元素
```

char buf[256]  (栈 / 结构体内部)
┌────────────────────────────────────────┐
│ 'a' │ 'b' │ 'c' │ ... (共 256 字节)   │  <- sizeof(buf) == 256
└────────────────────────────────────────┘

std::string s; s.resize(256);
[栈 / 结构体内部: sizeof(s) == 32 字节]          [堆内存: 实际 256+ 字节]
┌──────────────────────────────────────┐         ┌────────────────────────┐
│ ptr / sso_buf │ size=256 │ cap=... │ ──────> │ 'a' │ 'b' │ 'c' │ ...  │
└──────────────────────────────────────┘         └────────────────────────┘

## 2026-11-24

ASAN 不需要 `-fno-santize-recover=all` 而 UBSAN 需要的原因是，前者默认检测到就报错直接退出，而后者则默认恢复继续运行。

target_complie_opiton 和 target_link_option 中的内容不同，是因为有些行为只在编译，有些行为只在链接。

如果感到一个函数的返回值拿不定主意，思考是不是它需要拆分为职责更清晰的几个函数。

### auto

最简单的场景：auto x {a}，a 的类型是 T，auto 作为类型占位符 U 进行推导得到类型 T，因此 x 的推导类型是 T，然后用 a 的值进行初始化。如果 a 有 const 类型限定符，auto 会去掉它。

再考虑这样的场景：auto x {&a}，&a 的类型是 T\*，auto 作为类型占位符 U 对 T\* 进行推导得到 &T，因此 x 的类型是 T\*，用 &a 的值初始化 x 后，x 指向 a。显然 a 的 const 应该保留，因为创建一个新的指向 a 变量 x 不应改变 a。&a 的 const 则默认去除，因为 auto 希望能得到一个可供修改的独立变量 x。这就是 auto 保留底层 const 去除顶层 const 的原因。

为了得到有自己内存和生命周期的独立对象，auto 会去掉推导类型中的引用。

对于 auto\*，举个例子：auto* x {&a}，这里 &a 的类型是 T\*，而 auto\* 会被看作类型占位符 U\* 对 T\* 进行推导，得到 auto 为 T，因此 x 的推导类型为 T*，然后用 &a 的值来初始化 x。auto\* 只接受指针初始化器。

对于 auto& 也类似：a 的类型是 T，用 U& 去匹配 T，得到 U 为 T，所以 x 的类型是 T&。这里的 T& 必须去掉引用后才能推导 U。因此 x 的类型为 auto& 即 T&，绑定到 a 上。若 T 存在 const 则会被保留

打个比方：
- auto x = s; —— 自己复印一份新盒子
- auto& r = s; —— 给原盒子贴一张新名字标签
- auto* p = &s; —— 抄写一张写着“盒子门牌号”的纸条
