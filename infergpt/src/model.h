#pragma once

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <map>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "matrix.h"
#include "layers.h"

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
class Model {
 public:
  explicit Model(const HyperParameters& hyper_params) : hyper_params_(hyper_params),
    wte_(hyper_params.vocabulary_size, hyper_params.EmbeddingSize()),
    wpe_(hyper_params.context_limit, hyper_params.EmbeddingSize()),
    ln_f(hyper_params.EmbeddingSize()),
    transformers_(hyper_params.layers, Transformer<T>{hyper_params.EmbeddingSize(), hyper_params.attention_heads}) {
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

  // Loads a list of token ids from a text file.
  static RowVector<TokenId> LoadTokenIds(const std::filesystem::path& path) {
    std::vector<TokenId> tokens;
    std::string line;
    for (std::ifstream f{path}; std::getline(f, line); )
      tokens.push_back(std::atoi(line.c_str()));
    return tokens;
  }

  Matrix<T> Forward(std::span<const TokenId> inputs) const {
    Matrix<T> x = Embed(inputs);
    for (const auto& transformer : transformers_)
      x = transformer(std::move(x));
    return ln_f(x) * Transpose(wte_);
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
  Matrix<T> wte_;
  Matrix<T> wpe_;
  LayerNorm<T> ln_f;
  std::vector<Transformer<T>> transformers_;
};

// Splits a string like "name: 42" or "name, 42".
inline std::tuple<std::string, int> Split(const std::string& line) {
  std::string separators = ":, ";
  const auto first_separator = line.find_first_of(separators);
  const auto last_separator = line.find_last_of(separators);
  return {line.substr(0, first_separator),
    std::atoi(line.substr(last_separator + 1).c_str())};
}

// Loads a set of hyperparameters from a text file.
inline HyperParameters LoadHyperParameters(const std::filesystem::path& path) {
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
