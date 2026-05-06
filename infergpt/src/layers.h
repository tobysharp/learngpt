#pragma once

#include <cmath>
#include <limits>
#include <numbers>
#include <string>

#include "algebra.h"
#include "matrix.h"

// A simple affine transformation layer with weights and bias, computing Y = X * W + B.
template <typename T>
struct Affine {
  Matrix<T> weights_T;
  RowVector<T> bias;
  explicit Affine(int rows, int cols) : weights_T(cols, rows), bias(cols) {}
  void Load(const std::string& stem) {
    Matrix<T> weights(weights_T.Columns(), weights_T.Rows());
    weights.Load(stem + "_w.bin");
    weights_T = Transpose(weights);
    bias.Load(stem + "_b.bin");
  }
  Matrix<T> operator()(const Matrix<T>& x) const {
    auto y = MatMul_XYT(x, weights_T);
    y += BroadcastToRows(bias, x.Rows());
    return y;
  }
  RowVector<T> operator()(const RowVector<T>& x) const {
    auto y = MatMul_XYT(x, weights_T);
    y += bias;
    return y;
  }
};

// Layer normalization layer, normalizing each row to have zero mean and unit variance, then applying a learned scale and bias.
template <typename T>
struct LayerNorm {
  RowVector<T> g, b;
  
  explicit LayerNorm(int cols) : g(cols), b(cols) {}
  void Load(const std::string& stem) {
    g.Load(stem + "_g.bin");
    b.Load(stem + "_b.bin");
  }
  template <WritableTensor X>
  X operator()(X x) const {
    LayerNormInPlace(x);
    return x;
  }
  template <IsTensor X> requires (!WritableTensor<X>)
  ValueTensor<X> operator()(const X& x) const {
    ValueTensor<X> copy = x;
    LayerNormInPlace(copy);
    return copy;
  }
 private:
  template <IsMatrix X> requires (!IsVector<X>)
  void LayerNormInPlace(X&& x) const {
    for (int i = 0; i < x.Rows(); ++i)
      LayerNormInPlace(Row(x, i));
  }
  template <IsVector V>
  void LayerNormInPlace(V&& row) const {
    constexpr T eps = T{1e-5};
    auto [mean, variance] = MeanAndVariance(row);
    const auto scale = T{1} / std::sqrt(variance + eps);
    for (int j = 0; j < row.Size(); ++j)
      row(j) = (row(j) - mean) * scale * g(j) + b(j);
  }
};

// Multi-head self-attention layer, computing attention in parallel across multiple heads, then projecting the concatenated output.
template <typename T>
struct MultiHeadAttention {
  explicit MultiHeadAttention(int embedding_size, int heads) : c_attn_(embedding_size, 3 * embedding_size), c_proj_(embedding_size, embedding_size), heads_(heads) {}

  void Load(const std::string& stem ) {
    c_attn_.Load(stem + "_c_attn");
    c_proj_.Load(stem + "_c_proj");
  }

  using KVCache = Matrix<T>;

  std::tuple<Matrix<T>, KVCache> Prefill(Matrix<T> x) const {
    const int sequence_size = x.Rows();
    const int embedding_size = x.Columns();
    const auto qkv = c_attn_(x);  // QKV projection.
    const int dims_per_head = embedding_size / heads_;
    for (int i = 0; i < heads_; ++i) {
      auto q = Block(qkv, 0, 0 * embedding_size + i * dims_per_head, sequence_size, dims_per_head);
      auto k = Block(qkv, 0, 1 * embedding_size + i * dims_per_head, sequence_size, dims_per_head);
      auto v = Block(qkv, 0, 2 * embedding_size + i * dims_per_head, sequence_size, dims_per_head);
      Block(x, 0, dims_per_head * i, sequence_size, dims_per_head) = CausalAttention(q, k, v);
    }
    KVCache cache = Block(qkv, 0, embedding_size, qkv.Rows(), 2 * embedding_size);
    return {c_proj_(x), cache};
  }

  RowVector<T> Decode(RowVector<T> x, KVCache* cache) const {
    const int embedding_size = x.Columns();
    const auto qkv = c_attn_(x);  // QKV projection.
    cache->AddRow(SubVector(qkv, embedding_size, 2 * embedding_size));
  
    const int dims_per_head = embedding_size / heads_;
    //ParallelFor(0, heads_, [&](int i) {
    for (int i = 0; i < heads_; ++i) {
      auto q = SubVector(qkv, i * dims_per_head, dims_per_head);
      auto k = Block(*cache, 0, i * dims_per_head, cache->Rows(), dims_per_head);
      auto v = Block(*cache, 0, embedding_size + i * dims_per_head, cache->Rows(), dims_per_head);
      SubVector(x, dims_per_head * i, dims_per_head) = CausalAttention(q, k, v);
    }
    //);
    return c_proj_(x);
  }

  Matrix<T> operator()(Matrix<T> x) const {
    return std::get<0>(Prefill(std::move(x)));
  }

 private:
  // Applies the softmax function to the first `count` elements of `x` in place.
  static void SoftmaxInPlace(RowVector<T>& x, int count) {
    T x_max = std::numeric_limits<T>::lowest();
    for (int i = 0; i < count; ++i)
      x_max = std::max(x_max, x(i));
    T sum = T{0};
    for (int i = 0; i < count; ++i) {
      x(i) = std::exp(x(i) - x_max);
      sum += x(i);
    }
    const T scale = T{1} / sum;
    for (int i = 0; i < count; ++i)
      x(i) *= scale;
  }

  // Computes the causal attention of q, k, v, where each row i of the output is the attention of q.row(i) with k.rows(0..i) and v.rows(0..i).
  template <IsMatrix M>
  static auto CausalAttention(const M& q, const M& k, const M& v) {
    using R = typename std::remove_cvref_t<M>;
    static_assert(std::is_same_v<T, typename R::Scalar>);  

    Matrix<T> out{q.Rows(), v.Columns()};
    RowVector<T> row{k.Rows()};
    const T scale = T{1} / std::sqrt(static_cast<T>(q.Columns()));
    for (int i = 0; i < q.Rows(); ++i) {
      for (int j = 0; j <= i; ++j)
        row(j) = Dot(Row(q, i), Row(k, j)) * scale;

      SoftmaxInPlace(row, i + 1);

      T* dst = out.RowData(i);
      std::fill(dst, dst + out.Columns(), T{0});
      for (int ii = 0; ii <= i; ++ii) {
        const T* vdata = v.RowData(ii);
        const T lhs = row(ii);
        for (int j = 0; j < v.Columns(); ++j)
          dst[j] += lhs * vdata[j];
      }
    }
    return out;
  }

  template <IsVector Qi, IsMatrix KV>
  static RowVector<T> CausalAttention(const Qi& qi, const KV& k, const KV& v) {
    const T scale = T{1} / std::sqrt(static_cast<T>(qi.Size()));
    RowVector<T> row = MatMul_XYT(qi * scale, k);
    SoftmaxInPlace(row, row.Size());
    return row * v;
  }

  Affine<T> c_attn_;
  Affine<T> c_proj_;
  const int heads_;
};

// A feedforward MLP layer, applying a non-linear activation between two affine transformations.
template <typename T>
struct MultiLayerPerceptron {
  Affine<T> c_fc;
  Affine<T> c_proj;
  explicit MultiLayerPerceptron(int embedding_size) : c_fc(embedding_size, 4 * embedding_size), c_proj(4 * embedding_size, embedding_size) {}
  void Load(const std::string& stem) {
    c_fc.Load(stem + "_c_fc");
    c_proj.Load(stem + "_c_proj");
  }
  template <WritableTensor X>
  ValueTensor<X> operator()(X x) const {
    return c_proj(Gelu(c_fc(x)));
  }
 private:
  // The GELU activation function, approximating x * sigmoid(1.702 * x).
  template <WritableTensor X>
  static ValueTensor<X> Gelu(X x) {
    const T sqrt_two_over_pi = std::sqrt(T{2} / std::numbers::pi_v<T>);
    return Transform(std::move(x), [sqrt_two_over_pi](T value) {
      return T{0.5} * value * (T{1} + std::tanh(sqrt_two_over_pi * (value + static_cast<T>(0.044715) * value * value * value)));
    });
  }
};

// A transformer layer, consisting of a multi-head self-attention layer followed by a feedforward MLP layer, with layer normalization and residual connections.
template <typename T>
struct Transformer {
  using Cache = typename MultiHeadAttention<T>::KVCache;

  MultiHeadAttention<T> attention;
  MultiLayerPerceptron<T> mlp;
  LayerNorm<T> ln_1, ln_2;
  explicit Transformer(int embedding_size, int heads) : ln_1(embedding_size), ln_2(embedding_size),
                                        attention(embedding_size, heads), mlp(embedding_size) {}
  void Load(const std::string& stem) {
    attention.Load(stem + "_attn");
    mlp.Load(stem + "_mlp");
    ln_1.Load(stem + "_ln_1");
    ln_2.Load(stem + "_ln_2");
  }
  
  std::tuple<Matrix<T>, Cache> Prefill(Matrix<T> x) const {
    const auto [attn, cache] = attention.Prefill(ln_1(x));
    x += attn;
    x += mlp(ln_2(x));
    return {x, cache};
  }

  RowVector<T> Decode(RowVector<T> x, Cache* cache) const {
    x += attention.Decode(ln_1(x), cache);
    x += mlp(ln_2(x));
    return x;
  }

  Matrix<T> operator()(Matrix<T> x) const {
    return std::get<0>(Prefill(std::move(x)));
  }
};
