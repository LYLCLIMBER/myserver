#include <limits>
#include <optional>
#include <string>
#include <llama.h>
#include <iostream>
#include <vector>
#include <string_view>
#include <cstdint>

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
void print_model_desc(const llama_model* model)
{
  char buffer[256] {};
  const int32_t length {llama_model_desc(model, buffer, sizeof(buffer))};

  if (length < 0) {
    std::cerr << __func__ << ": error: 获取模型描述失败\n";
    return;
  }
  // desc 函数底层使用 snprintf，返回的输出字节数不包括最后终止字符串的 \0
  else if (static_cast<size_t>(length) >= sizeof(buffer)) {
    std::cerr << __func__ << ": error: 字符缓存区不足，需要" << static_cast<size_t>(length) + 1 << "字节\n";
    return;
  }
  // std::cout 会从开头读到 \0，没有 \0 会造成未定义行为
  std::cout << buffer << "\n";
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
} // namespace

int main()
{
  // 初始化配置，目前暂时写死
  const std::string model_path {"/home/liuyulong/Codes/project/models/qwen2.5-3b-instruct-q4_k_m.gguf"};
  const std::string prompt {"你好"};
  const int ngl {99};
  const int n_predict {32};

  if (model_path.empty()) {
    return 1;
  }

  // 只打印错误日志
  // TODO: logger 是全局状态，"NOT thread safe"
  llama_log_set(log_callback, nullptr);

  llama_backend_init();

  // 初始化模型配置和实例
  auto model_params {llama_model_default_params()};
  model_params.n_gpu_layers = ngl;

  auto* model {llama_model_load_from_file(model_path.c_str(), model_params)};
  if (model == nullptr) {
    std::cerr << __func__ << ": error: unable to load model\n";
    llama_backend_free();
    return 1;
  }
  print_model_desc(model);

  // 分词
  const auto* vocab {llama_model_get_vocab(model)};
  const auto tokenize_result {tokenize(vocab, prompt)};
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
  ctx_params.n_ctx = static_cast<uint32_t>(n_prompt + n_predict - 1);
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
  // 逐个 token 的输出循环
  llama_sampler_free(smpl);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
}
