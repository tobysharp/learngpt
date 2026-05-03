#pragma once

#include <cmath>
#include <limits>
#include <numbers>
#include <string>

#include "algebra.h"
#include "matrix.h"

// A simple affine transformation layer with weights and bias, computing Y = X * W + B.
template <typename T, typename RowSize, typename ColSize>
struct Affine {
  Matrix<T, RowSize, ColSize> weights;
  RowVector<T, ColSize> bias;
  explicit Affine(int rows, int cols) : weights(rows, cols), bias(cols) {}
  void Load(const std::string& stem) {
    weights.Load(stem + "_w.bin");
    bias.Load(stem + "_b.bin");
  }
  template <typename LhsRowSize>
  Matrix<T, LhsRowSize, ColSize> operator()(const Matrix<T, LhsRowSize, RowSize>& x) const {
    return x * weights + bias;
  }
};

// Layer normalization layer, normalizing each row to have zero mean and unit variance, then applying a learned scale and bias.
template <typename T, typename Size>
struct LayerNorm {
  RowVector<T, Size> g, b;
  explicit LayerNorm(int cols) : g(cols), b(cols) {}
  void Load(const std::string& stem) {
    g.Load(stem + "_g.bin");
    b.Load(stem + "_b.bin");
  }
  template <typename RowAxis, typename ColAxis>
  Matrix<T, RowAxis, ColAxis> operator()(Matrix<T, RowAxis, ColAxis> x) const {
    constexpr float eps = 1e-5f;
    return x; // TODO
  }
};

// Multi-head self-attention layer, computing attention in parallel across multiple heads, then projecting the concatenated output.
template <typename T, typename EmbedSize, typename EmbedSizeX3>
struct MultiHeadAttention {
  explicit MultiHeadAttention(int embedding_size, int heads) : c_attn_(embedding_size, 3 * embedding_size), c_proj_(embedding_size, embedding_size), heads_(heads) {}

  void Load(const std::string& stem ) {
    c_attn_.Load(stem + "_c_attn");
    c_proj_.Load(stem + "_c_proj");
  }

  template <typename SeqSize>
  Matrix<T, SeqSize, EmbedSize> operator()(Matrix<T, SeqSize, EmbedSize> x) const {
    const int sequence_size = x.Rows();
    const int embedding_size = x.Columns();
    const auto qkv = c_attn_(x);  // QKV projection.
    const int dims_per_head = embedding_size / heads_;
    struct PerHeadSize {};
    for (int i = 0; i < heads_; ++i) {
      auto q = Block<SeqSize, PerHeadSize>(qkv, 0, 0 * embedding_size + i * dims_per_head, sequence_size, dims_per_head);
      auto k = Block<SeqSize, PerHeadSize>(qkv, 0, 1 * embedding_size + i * dims_per_head, sequence_size, dims_per_head);
      auto v = Block<SeqSize, PerHeadSize>(qkv, 0, 2 * embedding_size + i * dims_per_head, sequence_size, dims_per_head);
      Block<SeqSize, PerHeadSize>(x, 0, dims_per_head * i, sequence_size, dims_per_head) = CausalAttention(q, k, v);
    }
    return c_proj_(x);
  }

 private:
  // Applies the softmax function to the first `count` elements of `x` in place.
  template <typename Size>
  static void SoftmaxInPlace(RowVector<T, Size>& x, int count) {
    T x_max = std::numeric_limits<T>::lowest();
    for (int i = 0; i < count; ++i)
      x_max = std::max(x_max, x[i]);
    T sum = T{0};
    for (int i = 0; i < count; ++i) {
      x[i] = std::exp(x[i] - x_max);
      sum += x[i];
    }
    const T scale = T{1} / sum;
    for (int i = 0; i < count; ++i)
      x[i] *= scale;
  }

  // Computes the causal attention of q, k, v, where each row i of the output is the attention of q.row(i) with k.rows(0..i) and v.rows(0..i).
  template <IsMatrix M>
  static auto CausalAttention(const M& q, const M& k, const M& v) {
    using R = typename std::remove_cvref_t<M>;
    static_assert(std::is_same_v<T, typename R::Scalar>);  
    using Rows = typename R::RowAxis;
    using Cols = typename R::ColAxis;

    Matrix<T, Rows, Cols> out{q.Rows(), v.Columns()};
    RowVector<T, Rows> row{k.Rows()};
    const T scale = T{1} / std::sqrt(static_cast<T>(q.Columns()));
    for (int i = 0; i < q.Rows(); ++i) {
      for (int j = 0; j <= i; ++j)
        row[j] = Dot(Row(q, i), Row(k, j)) * scale;
      SoftmaxInPlace(row, i + 1);
      // Dot of first (i+1) elements of `row` with v.column(j).
      for (int j = 0; j < v.Columns(); ++j)
        out(i, j) = Dot(row, Column(v, j), i + 1); 
    }
    return out;
  }

  Affine<T, EmbedSize, EmbedSizeX3> c_attn_;
  Affine<T, EmbedSize, EmbedSize> c_proj_;
  const int heads_;
};

// A feedforward MLP layer, applying a non-linear activation between two affine transformations.
template <typename T, typename EmbedSize, typename EmbedSizeX4>
struct MultiLayerPerceptron {
  Affine<T, EmbedSize, EmbedSizeX4> c_fc;
  Affine<T, EmbedSizeX4, EmbedSize> c_proj;
  explicit MultiLayerPerceptron(int embedding_size) : c_fc(embedding_size, 4 * embedding_size), c_proj(4 * embedding_size, embedding_size) {}
  void Load(const std::string& stem) {
    c_fc.Load(stem + "_c_fc");
    c_proj.Load(stem + "_c_proj");
  }
  template <typename SeqSize>
  Matrix<T, SeqSize, EmbedSize> operator()(Matrix<T, SeqSize, EmbedSize> x) const {
    return c_proj(Gelu(c_fc(x)));
  }
 private:
  // The GELU activation function, approximating x * sigmoid(1.702 * x).
  template <typename RowSize, typename ColSize>
  static Matrix<T, RowSize, ColSize> Gelu(Matrix<T, RowSize, ColSize> m) {
    const T sqrt_two_over_pi = std::sqrt(T{2} / std::numbers::pi_v<T>);
    return Transform(std::move(m), [sqrt_two_over_pi](T x) {
      return T{0.5} * x * (T{1} + std::tanh(sqrt_two_over_pi * (x + static_cast<T>(0.044715) * x * x * x)));
    });
  }
};

// A transformer layer, consisting of a multi-head self-attention layer followed by a feedforward MLP layer, with layer normalization and residual connections.
template <typename T, typename EmbedSize, typename EmbedSizeX3, typename EmbedSizeX4>
struct Transformer {
  MultiHeadAttention<T, EmbedSize, EmbedSizeX3> attention;
  MultiLayerPerceptron<T, EmbedSize, EmbedSizeX4> mlp;
  LayerNorm<T, EmbedSize> ln_1, ln_2;
  explicit Transformer(int embedding_size, int heads) : ln_1(embedding_size), ln_2(embedding_size),
                                        attention(embedding_size, heads), mlp(embedding_size) {}
  void Load(const std::string& stem) {
    attention.Load(stem + "_attn");
    mlp.Load(stem + "_mlp");
    ln_1.Load(stem + "_ln_1");
    ln_2.Load(stem + "_ln_2");
  }
  template <typename SeqSize>
  Matrix<T, SeqSize, EmbedSize> operator()(Matrix<T, SeqSize, EmbedSize> x) const {
    x += attention(ln_1(x));
    x += mlp(ln_2(x));
    return x;
  }
};
