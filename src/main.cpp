#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <llama.h>
#include <iostream>
#include <vector>
#include <string_view>
#include <cstdint>
#include <span>
#include <charconv>

// 防止 ASAN 检查 cuda 报错
#if defined(__SANITIZE_ADDRESS__)
#include <sanitizer/asan_interface.h>

extern "C" const char* __asan_default_options()
{
  return "protect_shadow_gap=0";
}
#endif

namespace
{
struct Config {
  std::string model_path {};
  std::string prompt {"你好"};
  int ngl {99};
  int n_predict {32};
};

void print_model_desc(const llama_model* model)
{
  char buffer[256] {};
  const int32_t length {llama_model_desc(model, buffer, sizeof(buffer))};

  if (length < 0) {
    std::cerr << __func__ << ": error: 获取模型描述失败\n";
    return;
  }
  // desc 函数底层使用 snprintf，返回的输出字节数不包括最后终止字符串的 \0，因此需要 =
  else if (static_cast<size_t>(length) >= sizeof(buffer)) {
    std::cerr << __func__ << ": error: 字符缓存区不足，需要" << static_cast<size_t>(length) + 1 << "字节\n";
    return;
  }
  // std::cout 会从开头读到 \0，没有 \0 会造成未定义行为
  std::cout << "模型描述符：" << buffer << "\n";
}

void log_callback(enum ggml_log_level level, const char* text, void* /* user_data */)
{
  // static：CONT 日志沿用上一条非 CONT 日志的决定
  static bool should_print {false};
  constexpr ggml_log_level target {GGML_LOG_LEVEL_ERROR};

  if (level != GGML_LOG_LEVEL_CONT) {
    should_print = (level >= target);
  }
  if (should_print) {
    std::cerr << text;
  }
}

std::optional<std::vector<llama_token>> tokenize(const llama_vocab* vocab, std::string_view text)
{
  // 获取 text tokenize 后的长度
  const size_t text_len {text.size()};
  if (text_len > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    std::cerr << __func__ << ": error: text 长度溢出\n";
    return std::nullopt;
  }
  // TODO: add_special 允许加入 BOS 和 EOS，parse_special 解析特殊字符串如 <|im_start|>，但可能引入提示词注入风险
  const int32_t neg_n_text {llama_tokenize(vocab, text.data(), static_cast<int32_t>(text_len), nullptr, 0, true, true)};
  if (neg_n_text > 0) {
    std::cerr << __func__ << ": error: text token 长度为负数\n";
    return std::nullopt;
  } else if (neg_n_text == INT32_MIN) {
    std::cerr << __func__ << ": error: text token 长度溢出\n";
    return std::nullopt;
  }
  const int32_t n_text {-neg_n_text};

  std::vector<llama_token> text_tokens(static_cast<size_t>(n_text));
  if (llama_tokenize(vocab, text.data(), static_cast<int32_t>(text_len), text_tokens.data(), n_text, true, true) !=
      n_text) {
    std::cerr << __func__ << ": error: tokenize 失败\n";
    return std::nullopt;
  }
  return text_tokens;
}

std::optional<std::string> token_to_piece(const llama_vocab* vocab, llama_token token)
{
  constexpr size_t init_length {128};
  std::string piece;
  piece.resize(init_length);
  const int32_t n {llama_token_to_piece(vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, true)};
  if (n >= 0) {
    piece.resize(static_cast<size_t>(n));
    return piece;
  }
  if (n == INT32_MIN) {
    std::cerr << __func__ << ": error: 词元转换文本溢出\n";
    return std::nullopt;
  }
  const int32_t needed {-n};
  piece.resize(static_cast<size_t>(needed));
  if (llama_token_to_piece(vocab, token, piece.data(), needed, 0, true) != needed) {
    std::cerr << __func__ << ": error: 词元转文本失败\n";
    return std::nullopt;
  }
  return piece;
}

std::optional<std::string> detokenize(const llama_vocab* vocab, const std::vector<llama_token>& tokens)
{
  std::string res;
  for (const llama_token token : tokens) {
    const auto piece {token_to_piece(vocab, token)};
    if (!piece) {
      return std::nullopt;
    }
    res += *piece;
  }
  return res;
}

// TODO: 这里暂时只是单序列
// 清空 batch，从 pos_0 开始连续填充 token 到 batch 中，连续设置位置编码；只设置输出最后一个 token 的 logits
bool batch_set_tokens(llama_batch_ext* batch, std::span<const llama_token> tokens, llama_pos pos_0)
{
  llama_batch_ext_clear(batch);
  if (tokens.empty()) {
    std::cerr << __func__ << ": error: 填充 tokens 为空\n";
    return false;
  }
  int32_t idx {};
  for (size_t i {0}; i < tokens.size(); ++i) {
    idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
    if (idx < 0) {
      std::cerr << __func__ << ": error: 填充 tokens 失败，返回" << idx << "\n";
      return false;
    }
    // tokenize 保证了 tokens 长度不会超过 INT32_MAX
    const llama_pos pos {pos_0 + static_cast<llama_pos>(i)};
    if (!llama_batch_ext_set_pos(batch, idx, &pos)) {
      std::cerr << __func__ << ": error: pos 设置失败\n";
      return false;
    }
  }
  if (!llama_batch_ext_set_output_logits(batch, idx, true)) {
    std::cerr << __func__ << ": error: output logits 设置失败\n";
    return false;
  }
  return true;
}

void print_usage(char** argv)
{
  std::cerr << "\nexample usage:\n";
  std::cerr << argv[0] << "    -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [-p prompt]\n\n";
}

std::optional<int32_t> string_to_int(std::string_view s)
{
  int32_t res {};
  auto [ptr, ec] {std::from_chars(s.data(), s.data() + s.size(), res)};
  if (ec == std::errc {}) {
    if (ptr == s.data() + s.size()) {
      return res;
    } else {
      std::cerr << __func__ << ": error: 这不是一个数字\n";
      return std::nullopt;
    }
  } else if (ec == std::errc::invalid_argument) {
    std::cerr << __func__ << ": error: 这不是一个数字\n";
    return std::nullopt;
  } else if (ec == std::errc::result_out_of_range) {
    std::cerr << __func__ << ": error: 数值溢出\n";
    return std::nullopt;
  } else {
    std::cerr << __func__ << ": error: 转换错误\n";
    return std::nullopt;
  }
}

std::optional<Config> parse_args(int argc, char* argv[])
{
  Config config {};
  int32_t i = 1;

  for (; i < argc; ++i) {
    // argv[i] 以空字符结尾，不用担心 sting_view 越界
    const std::string_view arg {argv[i]};
    // string_view 能够直接和字符串字面量比较
    if (arg == "-m") {
      if (i + 1 < argc) {
        config.model_path = argv[++i];
      } else {
        print_usage(argv);
        return std::nullopt;
      }
    } else if (arg == "-n") {
      if (i + 1 < argc) {
        auto n_opt {string_to_int(argv[++i])};
        if (!n_opt) {
          return std::nullopt;
        }
        if (*n_opt <= 0) {
          std::cerr << __func__ << ": error: n 只接受正整数\n";
          return std::nullopt;
        }
        config.n_predict = *n_opt;
      } else {
        print_usage(argv);
        return std::nullopt;
      }
    } else if (arg == "-ngl") {
      if (i + 1 < argc) {
        auto ngl_opt {string_to_int(argv[++i])};
        if (!ngl_opt) {
          return std::nullopt;
        }
        if (*ngl_opt < 0) {
          std::cerr << __func__ << ": error: ngl 不接受负数\n";
          return std::nullopt;
        }
        config.ngl = *ngl_opt;
      } else {
        print_usage(argv);
        return std::nullopt;
      }
    } else if (arg == "-p") {
      if (i + 1 < argc) {
        config.prompt = argv[++i];
      } else {
        print_usage(argv);
        return std::nullopt;
      }
    } else {
      print_usage(argv);
      return std::nullopt;
    }
  }
  if (config.model_path.empty()) {
    std::cerr << __func__ << ": error: 模型路径为空\n";
    return std::nullopt;
  }
  return config;
}
} // namespace

int main(int argc, char* argv[])
{
  // 解析命令行参数
  auto parse_res {parse_args(argc, argv)};
  if (!parse_res) {
    std::cerr << __func__ << ": error: 命令行参数解析失败\n";
    return 1;
  }
  const Config config {*parse_res};

  // 只打印错误日志
  // TODO: logger 是全局状态，"NOT thread safe"
  llama_log_set(log_callback, nullptr);

  llama_backend_init();

  // 初始化模型配置和实例
  auto model_params {llama_model_default_params()};
  model_params.n_gpu_layers = config.ngl;

  auto* model {llama_model_load_from_file(config.model_path.c_str(), model_params)};
  if (model == nullptr) {
    std::cerr << __func__ << ": error: unable to load model\n";
    llama_backend_free();
    return 1;
  }
  print_model_desc(model);

  if (llama_model_has_encoder(model)) {
    std::cerr << __func__ << ": error: 不支持有 encoder 的模型\n";
    llama_model_free(model);
    llama_backend_free();
    return 1;
  }

  // 分词
  const auto* vocab {llama_model_get_vocab(model)};
  const auto tokenize_result {tokenize(vocab, config.prompt)};
  if (!tokenize_result) {
    llama_model_free(model);
    llama_backend_free();
    return 1;
  }
  const std::vector<llama_token>& prompt_tokens {*tokenize_result};
  if (prompt_tokens.empty()) {
    std::cerr << __func__ << ": error: tokenize 结果为空\n";
    llama_model_free(model);
    llama_backend_free();
    return 1;
  }
  // tokenize 保证了 prompt_tokens 长度小于等于 INT32_MAX
  const int n_prompt {static_cast<int>(prompt_tokens.size())};

  // 打印 prompt
  const auto rec_prompt {detokenize(vocab, prompt_tokens)};
  if (!rec_prompt) {
    llama_model_free(model);
    llama_backend_free();
    return 1;
  }
  std::cerr << "prompt: " << *rec_prompt << "\n";

  // 初始化运行时配置和实例
  auto ctx_params {llama_context_default_params()};
  // TODO: 后续不应该让单次 prompt 长度动态决定 batch 大小
  ctx_params.n_batch = static_cast<uint32_t>(n_prompt);
  ctx_params.no_perf = false;
  // 最后一个 token 不送回上下文
  const int n_ctx_train {llama_model_n_ctx_train(model)};
  if (n_prompt > n_ctx_train - config.n_predict + 1) {
    std::cerr << __func__ << ": error: 上下文长度超出模型训练时的上下文长度 " << n_ctx_train << "\n";
    llama_model_free(model);
    llama_backend_free();
    return 1;
  }
  ctx_params.n_ctx = static_cast<uint32_t>(n_prompt + config.n_predict - 1);
  auto* ctx {llama_init_from_model(model, ctx_params)};
  if (ctx == nullptr) {
    llama_model_free(model);
    llama_backend_free();
    std::cerr << __func__ << ": error: 运行时实例创建失败\n";
    return 1;
  }

  // 初始化采样器
  auto sparams {llama_sampler_chain_default_params()};
  sparams.no_perf = false;
  auto* smpl {llama_sampler_chain_init(sparams)};
  llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

  // 准备第一个 batch 输入
  llama_batch_ext* batch {llama_batch_ext_init(ctx)};
  if (batch == nullptr) {
    std::cerr << __func__ << ": error: batch 无效\n";
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 1;
  }
  if (!batch_set_tokens(batch, prompt_tokens, 0)) {
    llama_batch_ext_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 1;
  }

  // 循环输出 token

  int32_t n_tokens {n_prompt}; // 本次循环要处理的 token 数量，第一轮处理 prompt
  for (int32_t n_pos = 0; n_pos + n_tokens < n_prompt + config.n_predict;) {
    const int32_t ret {llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)};
    if (ret != 0) {
      std::cerr << __func__ << ": error: 推理失败，ret =" << ret << "\n";
      llama_batch_ext_free(batch);
      llama_sampler_free(smpl);
      llama_free(ctx);
      llama_model_free(model);
      llama_backend_free();
      return 1;
    }
    n_pos += n_tokens;

    const llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
    if (llama_vocab_is_eog(vocab, new_token_id)) {
      break;
    }
    auto res = token_to_piece(vocab, new_token_id);
    if (!res) {
      llama_batch_ext_free(batch);
      llama_sampler_free(smpl);
      llama_free(ctx);
      llama_model_free(model);
      llama_backend_free();
      return 1;
    }
    // flush 的时间成本和模型 token 耗时相比可以忽略
    std::cout << *res << std::flush;

    if (!batch_set_tokens(batch, {&new_token_id, 1}, n_pos)) {
      llama_batch_ext_free(batch);
      llama_sampler_free(smpl);
      llama_free(ctx);
      llama_model_free(model);
      llama_backend_free();
      return 1;
    }
    n_tokens = 1;
  }

  std::cout << "\n";

  llama_batch_ext_free(batch);
  llama_sampler_free(smpl);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
