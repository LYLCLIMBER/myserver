#include <ggml.h>
#include <string>
#include <llama.h>
#include <iostream>

void print_model_desc(const llama_model* model)
{
  char buffer[256] {};
  const int32_t length = llama_model_desc(model, buffer, sizeof(buffer));

  if(length < 0) {
    std::cerr << __func__ << ": error: 获取模型描述失败\n";
    return;
  }
  // desc 函数底层使用 snprintf，返回的输出字节数不包括最后终止字符串的 \0
  else if (static_cast<size_t>(length) >= sizeof(buffer)) {
    std::cerr << __func__ << ": error: 字符缓存区不足，需要" 
      << static_cast<size_t>(length) + 1 << "字节\n";
    return;
  }
  // std::cout 会从开头读到 \0，没有 \0 会造成未定义行为
  std::cout << buffer << "\n";
}

void log_callback(enum ggml_log_level level, const char* text, void* /* user_data */)
{
  // static：CONT 日志沿用上一条非 CONT 日志的决定
  static bool should_print {false};
  constexpr ggml_log_level target = GGML_LOG_LEVEL_ERROR;

  if (level != GGML_LOG_LEVEL_CONT) {
    should_print = (level >= target);
  }
  if (should_print){
    std::cerr << text;
  }
}

int main ()
{
  // 初始化配置，目前暂时写死
  std::string model_path {"/home/liuyulong/Codes/project/models/qwen2.5-3b-instruct-q4_k_m.gguf"};
  std::string prompt {"你好"};
  int ngl {99};
  // int n_predict {32};

  if (model_path.empty()){
    return 1;
  }

  // 只打印错误日志
  // TODO: logger 是全局状态，"NOT thread safe"
  llama_log_set(log_callback , nullptr);

  llama_backend_init();

  // 初始化模型
  llama_model_params model_params {llama_model_default_params()};
  model_params.n_gpu_layers = ngl;

  llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
  if(model == nullptr){
    std::cerr << __func__ << ": error: unable to load model\n";
    llama_backend_free();
    return 1;
  }
  print_model_desc(model);

  // 分词


  // 初始化运行时配置
  // 初始化采样器
  // 准备第一个 batch 输入
  // 逐个 token 的输出循环
  llama_model_free(model);
  llama_backend_free();
}

