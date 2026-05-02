#include <cassert>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <span>
#include <sstream>
#include <tuple>
#include <vector>

using TokenId = int32_t;

struct HyperParameters {
  int vocabulary_size;        // The number of tokens in the vocabulary.
  int context_limit;          // The maximum input sequence length, in tokens.
  int attention_heads;        // The number of parallel attention heads per layer.
  int head_dimensions;        // The dimensionality of a token embedding vector per attention head.
  int layers;                 // The number of transformer layers, in series.

  int EmbeddingSize() const { return head_dimensions * attention_heads; }
};

template <typename T = float>
class Matrix {
 public:
  Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows * cols) {}
  Matrix(int rows, int cols, std::vector<T>&& data) : rows_(rows), cols_(cols), data_(std::move(data)) {}
  Matrix(const Matrix&) = default;
  Matrix(Matrix&&) = default;

  int Rows() const { return rows_; }
  int Columns() const { return cols_; }
  
  T* RowData(int row = 0) { return data_.data() + row * Columns(); }
  const T* RowData(int row = 0) const { return data_.data() + row * Columns(); }

  T* operator[](int row) { return RowData(row); }
  const T* operator[](int row) const { return RowData(row); }

  T& operator()(int row, int col) { return RowData(row)[col]; }
  const T& operator()(int row, int col) const { return RowData(row)[col]; }

  void Load(const std::filesystem::path& path) {
    std::ifstream f{path, std::ios::binary};
    for (int i = 0; i < rows_; ++i)
      f.read(reinterpret_cast<char*>(RowData(i)), sizeof(T) * cols_);
    if (!f) throw std::runtime_error("Load from file failed.");
  }

  template <std::integral I>
  Matrix GatherRows(std::span<const I> row_indices) const {
    Matrix result{std::ssize(row_indices), cols_};
    int out = 0;
    for (I index : row_indices)
      std::copy(RowData(index), RowData(index + 1), result[out++]);
    return result;
  }

  template <std::integral I>
  Matrix operator[](std::span<const I> row_indices) const {
    return GatherRows(row_indices);
  }

  Matrix Transpose() const {
    Matrix result{cols_, rows_};
    for (int i = 0; i < rows_; ++i)
      for (int j = 0; j < cols_; ++j)
        result(j, i) = *this(i, j);
    return result;
  }

 protected:
  int rows_, cols_;
  std::vector<T> data_;
};

// A row vector.
template <typename T>
class Vector : public Matrix<T> {
 public:
  explicit Vector(int cols) : Matrix<T>(1, cols) {}
  Vector(std::vector<T>&& data) : Matrix<T>(1, std::ssize(data), std::move(data)) {}
  Vector(const Vector&) = default;
  Vector(Vector&&) = default;

  int Size() const { return this->Columns(); }

  T& operator[](int index) { return this->RowData()[index]; }
  const T& operator[](int index) const { return this->RowData()[index]; }

  operator std::span<T>() { return this->data_; }
  operator std::span<const T>() const { return this->data_; }
};

template <typename T>
Matrix<T> Gelu(const Matrix<T>& x) {
  // TODO
  return x;
}

template <typename T>
struct Affine {
  Matrix<T> weights;
  Vector<T> bias;
  explicit Affine(int rows, int cols) : weights(rows, cols), bias(cols) {}
  void Load(const std::string& stem) {
    weights.Load(stem + "_w.bin");
    bias.Load(stem + "_b.bin");
  }
  Matrix<T> operator()(const Matrix<T>& x) {
    return x * weights + bias;
  }
};

template <typename T>
struct LayerNorm {
  Vector<T> g, b;
  explicit LayerNorm(int cols) : g(cols), b(cols) {}
  void Load(const std::string& stem) {
    g.Load(stem + "_g.bin");
    b.Load(stem + "_b.bin");
  }
  Matrix<T> operator()(Matrix<T> x) {
    constexpr float eps = 1e-5f;
  }
};

template <typename T>
struct Attention {
  Affine<T> c_attn, c_proj;
  explicit Attention(int embedding_size) : c_attn(embedding_size, 3 * embedding_size), c_proj(embedding_size, embedding_size) {}
  void Load(const std::string& stem ) {
    c_attn.Load(stem + "_c_attn");
    c_proj.Load(stem + "_c_proj");
  }
  Matrix<T> operator()(Matrix<T> x) const {
    x = c_attn(x);
    // TODO: Multi-head attention
    return c_proj(x);
  }
};

template <typename T>
struct MultiLayerPerceptron {
  Affine<T> c_fc, c_proj;
  explicit MultiLayerPerceptron(int embedding_size) : c_fc(embedding_size, 4 * embedding_size), c_proj(4 * embedding_size, embedding_size) {}
  void Load(const std::string& stem) {
    c_fc.Load(stem + "_c_fc");
    c_proj.Load(stem + "_c_proj");
  }
  Matrix<T> operator()(const Matrix<T>& x) const {
    return c_proj(Gelu(c_fc(x)));
  }
};

template <typename T>
struct Transformer {
  Attention<T> attention;
  MultiLayerPerceptron<T> mlp;
  LayerNorm<T> ln_1, ln_2;
  explicit Transformer(int embedding_size) : ln_1(embedding_size), ln_2(embedding_size),
                                        attention(embedding_size), mlp(embedding_size) {}
  void Load(const std::string& stem) {
    attention.Load(stem + "_attn");
    mlp.Load(stem + "_mlp");
    ln_1.Load(stem + "_ln_1");
    ln_2.Load(stem + "_ln_2");
  }
  Matrix<T> operator()(Matrix<T> x) const {
    x += attention(ln_1(x));
    x += mlp(ln_2(x));
    return x;
  }
};

template <typename T = float>
class Model {
 public:
  explicit Model(const HyperParameters& hyper_params) : hyper_params_(hyper_params),
    wte_(hyper_params.vocabulary_size, hyper_params.EmbeddingSize()),
    wpe_(hyper_params.context_limit, hyper_params.EmbeddingSize()),
    ln_f(hyper_params.EmbeddingSize()),
    transformers_(hyper_params.layers, Transformer<T>{hyper_params.EmbeddingSize()}) {
  }

  static Model Load(const HyperParameters& hyper_params, const std::filesystem::path& dir) {
    Model model{hyper_params};
    model.wte_.Load(dir / "wte.bin");
    model.wpe_.Load(dir / "wpe.bin");
    model.ln_f.g.Load(dir / "ln_f_g.bin");
    model.ln_f.b.Load(dir / "ln_f_b.bin");
    for (int i = 0; i < std::ssize(model.transformers_); ++i)
      model.transformers_[i].Load(dir / (std::ostringstream{} << "blocks_" << i).str());
    return model;
  }

  Matrix<T> Forward(std::span<const TokenId> inputs) const {
    Matrix<T> x = Embed(inputs);
    for (const auto& transformer : transformers_)
      x = transformer(x);
    return ln_f(x) * wte_.Transpose();
  }

 private:
  Matrix<T> Embed(std::span<const TokenId> inputs) const {
    Matrix<T> result{static_cast<int>(std::ssize(inputs)), wte_.Columns()};
    for (int i = 0; i < std::ssize(inputs); ++i) {
      const float* token = wte_[inputs[i]];
      const float* pos = wpe_[i];
      float* dst = result[i];
      for (int col = 0; col < wte_.Columns(); ++col)
        dst[col] = token[col] + pos[col];
    }
    return result;
  }

  HyperParameters hyper_params_;
  Matrix<T> wte_, wpe_;
  LayerNorm<T> ln_f;
  std::vector<Transformer<T>> transformers_;
};

// Splits a string like "name: 42" or "name, 42".
std::tuple<std::string, int> Split(const std::string& line) {
  std::string separators = ":, ";
  const auto first_separator = line.find_first_of(separators);
  const auto last_separator = line.find_last_of(separators);
  return {line.substr(0, first_separator),
    std::atoi(line.substr(last_separator + 1).c_str())};
}


// Loads a set of hyperparameters from a text file.
HyperParameters LoadHyperParameters(const std::filesystem::path& path) {
  std::map<std::string, int> dict;
  std::string line;
  for (std::ifstream f{path}; std::getline(f, line); ) {
    const auto [name, value] = Split(line);
    dict[name] = value;
  }
  assert(dict["n_embd"] % dict["n_head"] == 0);
  return HyperParameters {
    .vocabulary_size = dict["n_vocab"],
    .context_limit = dict["n_ctx"],
    .attention_heads = dict["n_head"],
    .head_dimensions = dict["n_embd"] / dict["n_head"],
    .layers = dict["n_layer"]
  };
}

// Loads a list of token ids from a text file.
Vector<TokenId> LoadTokenIds(const std::filesystem::path& path) {
  std::vector<TokenId> tokens;
  std::string line;
  for (std::ifstream f{path}; std::getline(f, line); )
    tokens.push_back(std::atoi(line.c_str()));
  return tokens;
}

int main() {
  constexpr int tokens_to_generate = 40;

  const auto root = std::filesystem::path(__FILE__).parent_path() / "../..";
  const auto model_dir = root / "models/124M";

  const auto hyper_parameters = LoadHyperParameters(model_dir / "hparams.txt");
  const auto prompt = LoadTokenIds(root / "input.txt");
  
  assert(prompt.Size() + tokens_to_generate < hyper_parameters.context_limit);

  const auto model = Model<float>::Load(hyper_parameters, model_dir);

  std::cout << "Done" << std::endl;
}
