#pragma once

#include <cmath>
#include <limits>
#include <string>

#include "algebra.h"
#include "matrix.h"

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
struct MultiHeadAttention {
  explicit MultiHeadAttention(int embedding_size, int heads) : c_attn_(embedding_size, 3 * embedding_size), c_proj_(embedding_size, embedding_size), heads_(heads) {}

  void Load(const std::string& stem ) {
    c_attn_.Load(stem + "_c_attn");
    c_proj_.Load(stem + "_c_proj");
  }

  Matrix<T> operator()(Matrix<T> x) const {
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
    return c_proj_(x);
  }

 private:
  static void SoftmaxInPlace(Vector<T>& x, int count) {
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

  template <IsMatrix M>
  static Matrix<T> CausalAttention(const M& q, const M& k, const M& v) {
    static_assert(std::is_same_v<T, typename std::remove_cvref_t<M>::Scalar>);  

    Matrix<T> out{q.Rows(), v.Columns()};
    Vector<T> row{k.Rows()};
    const T scale = T{1} / std::sqrt(static_cast<T>(q.Columns()));
    for (int i = 0; i < q.Rows(); ++i) {
      for (int j = 0; j <= i; ++j)
        row[j] = RowDotRow(q, i, k, j) * scale;
      SoftmaxInPlace(row, i + 1);
      for (int j = 0; j < v.Columns(); ++j) {
        // Dot of first (i+1) elements of `row` with v.column(j).
        out(i, j) = RowDotColumn(row, 0, v, j, i + 1); 
      }
    }
    return out;
  }

  Affine<T> c_attn_, c_proj_;
  const int heads_;
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
  Matrix<T> operator()(Matrix<T> x) const {
    x += attention(ln_1(x));
    x += mlp(ln_2(x));
    return x;
  }
};
