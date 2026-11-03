# Tokenizer

Tokenizer提供了一个通用的分词器接口，支持多种后端（如Llama等），用于文本的分词（Tokenize）、反分词（Decode）以及词表信息的获取。该接口适用于自然语言处理任务中模型输入输出的编码与解码

## 1. 编译说明

支持在不同目标平台（Linux、Android、RISC-V、macOS）进行编译

### 1.1 编译脚本说明

build.sh支持的参数选项如下：

| 选项 | 说明                                             | 示例         |
| ---- | ------------------------------------------------ | ------------ |
| `-s` | 目标系统平台（`linux`、`android`、`riscv64`、`macos`） | `-s linux`   |
| `-a` | 目标架构（macOS：`arm64`）                    | `-a aarch64` |
| `-n` | SDK 名称（默认：`tokenizer`）                    | `-n my_sdk`  |
| `-b` | 编译类型（`Debug`、`Release`、`RelWithDebInfo`） | `-b Debug`   |
| `-m` | 启用ASAN（选填）                                 | `-m`         |

### 1.2 环境设置脚本

根据 `-s` 指定的目标系统，脚本将自动加载以下环境脚本之一：

- Linux 平台：`env_linux.sh`
- Android 平台：`env_android.sh`
- RISC-V 平台：`env_riscv64.sh`
- macOS 平台：`env_macos.sh`（使用系统 clang，无需额外交叉工具链）

请确保这些脚本存在且正确设置如下变量：

- `C_COMPILER` 和 `CXX_COMPILER`
- 相关工具链路径（例如 `ANDROID_NDK_PATH`）

编译工具版本要求：

* c++：要求C++11及以上版本
* cmake：要求3.14及以上版本

### 1.3 编译示例

编译 **Linux / aarch64**：

```
./build.sh -s linux -a aarch64 -b Release
```

编译 **Android / arm64-v8a**：

```
./build.sh -s android -a arm64-v8a -b Release
```

编译 **RISC-V 平台**：

```
./build.sh -s riscv64 -a generic -b Release
```

编译 **macOS / arm64**（Apple Silicon）：

```
./build.sh -s macos -a arm64 -b Release
```


启用 **ASAN**：

```
./build.sh -s linux -a aarch64 -b Debug -m
```

### 1.4 编译目录说明

脚本执行后将：

- 生成构建文件至：
   `./build/build_<sdk>_<system>_<arch>_<build_type>/`
- 安装构建产物至：
   `./install/<sdk>_<system>_<arch>/`

## 2. 代码接口说明

### 2.1 Tokenizer

```c++
/**
 * @brief 构造函数，从分词器文件中提取词表等信息，初始化分词器
 * @param type 分词器后端类型（如Llama等）
 * @param tokenizer_path 分词器文件路径（文件通常为GGUF格式）
 */
Tokenizer(TokenizerBackendType type, const char* tokenizer_path);

/**
 * @brief 构造函数，从分词器文件内存中提取词表等信息，初始化分词器
 * @param type 分词器后端类型（如Llama等）
 * @param tokenizer_buffer 分词器文件内存指针
 * @param buffer_size 分词器文件内存的大小
 */
Tokenizer(TokenizerBackendType type, const void * tokenizer_buffer, size_t buffer_size);
```

### 2.2 ~Tokenizer

```c++
/**
 * @brief 析构函数，释放分词器相关资源
 */
~Tokenizer();
```

### 2.3 GetVocabInfo

```c++
/**
 * @brief 获取词表信息
 * @param info 用于存储词表信息的结构体指针
 * @return 获取成功返回true，否则返回false
 */
bool GetVocabInfo(VocabInfo* info);
```

### 2.4 Tokenize

```c++
/**
 * @brief 将输入文本分词为token序列
 * @param text 输入文本指针
 * @param text_len 输入文本长度
 * @param tokens 输出token数组指针
 * @param n_tokens_max tokens数组的最大容量
 * @return 实际分词得到的token数量，失败返回负值
 */
int Tokenize(const char* text, int32_t text_len, int32_t* tokens, int32_t n_tokens_max);
```

### 2.5 TokenToPiece

```c++
 /**
  * @brief 将单个token转换为对应的字符串片段（piece）
  * @param token 单个token
  * @return 对应的字符串片段
  */
std::string TokenToPiece(int32_t token);
```

### 2.6 Decode

```c++
/**
 * @brief 将token数组还原为原始字符串
 * @param tokens token数组指针
 * @param n_tokens token数组长度
 * @return 还原后的字符串
 */
std::string Decode(int32_t* tokens, int32_t n_tokens);
```

## 3. 示例程序说明

示例代码的路径为：

```
./demo/tokenize_demo.cpp
```

执行编译后将生成可执行文件 tokenize_demo

tokenize_demo程序的基本使用方法：

```
./tokenize_demo -t <tokenizer_path> -p <prompt>
```

tokenize_demo程序输出示例：

```
./tokenize_demo -t ./Llama-2-hf-vocab.gguf -p "The weather is nice today"

llama_model_loader: loaded meta data with 30 key-value pairs and 0 tensors from ./Llama-2-hf-vocab.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Llama 2 7b Hf
llama_model_loader: - kv   3:                           general.finetune str              = hf
llama_model_loader: - kv   4:                           general.basename str              = Llama-2
llama_model_loader: - kv   5:                         general.size_label str              = 7B
llama_model_loader: - kv   6:                               general.tags arr[str,6]       = ["facebook", "meta", "pytorch", "llam...
llama_model_loader: - kv   7:                          general.languages arr[str,1]       = ["en"]
llama_model_loader: - kv   8:                          llama.block_count u32              = 32
llama_model_loader: - kv   9:                       llama.context_length u32              = 4096
llama_model_loader: - kv  10:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv  11:                  llama.feed_forward_length u32              = 11008
llama_model_loader: - kv  12:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv  13:              llama.attention.head_count_kv u32              = 32
llama_model_loader: - kv  14:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  15:                          general.file_type u32              = 1
llama_model_loader: - kv  16:                           llama.vocab_size u32              = 32000
llama_model_loader: - kv  17:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv  18:               general.quantization_version u32              = 2
llama_model_loader: - kv  19:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  20:                         tokenizer.ggml.pre str              = default
llama_model_loader: - kv  21:                      tokenizer.ggml.tokens arr[str,32000]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  22:                      tokenizer.ggml.scores arr[f32,32000]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  23:                  tokenizer.ggml.token_type arr[i32,32000]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv  24:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  25:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  26:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv  27:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  28:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  29:               tokenizer.ggml.add_eos_token bool             = false
init_tokenizer: initializing tokenizer for type 1
load: control token:      2 '</s>' is not marked as EOG
load: control token:      1 '<s>' is not marked as EOG
load: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
load: special tokens cache size = 3
load: token to piece cache size = 0.1684 MB
print_info: vocab type       = SPM
print_info: n_vocab          = 32000
print_info: n_merges         = 0
print_info: BOS token        = 1 '<s>'
print_info: EOS token        = 2 '</s>'
print_info: UNK token        = 0 '<unk>'
print_info: PAD token        = 0 '<unk>'
print_info: LF token         = 13 '<0x0A>'
print_info: EOG token        = 2 '</s>'
print_info: max token length = 48
vocab_info: vocab_size=32000 special_bos_id=1 special_eos_id=2
     1 -> ''
   450 -> ' The'
 14826 -> ' weather'
   338 -> ' is'
  7575 -> ' nice'
  9826 -> ' today'
Decode:  The weather is nice today
Total number of tokens: 6
```

