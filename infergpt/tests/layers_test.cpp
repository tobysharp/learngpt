#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <string>
#include <vector>

#include "layers.h"

namespace {

template <typename T>
Matrix<T> MakeMatrix(int rows, int cols, std::initializer_list<T> values) {
  assert(static_cast<int>(values.size()) == rows * cols);
  return Matrix<T>{rows, cols, std::vector<T>{values}};
}

template <typename T>
void FillRowVector(RowVector<T>* vector, std::initializer_list<T> values) {
  assert(vector->Size() == static_cast<int>(values.size()));
  int index = 0;
  for (const T value : values)
    (*vector)[index++] = value;
}

template <typename T>
void WriteMatrixFile(const std::filesystem::path& path, const Matrix<T>& matrix) {
  std::ofstream out(path, std::ios::binary);
  assert(out.good());
  for (int i = 0; i < matrix.Rows(); ++i)
    out.write(reinterpret_cast<const char*>(matrix[i]), sizeof(T) * matrix.Columns());
  assert(out.good());
}

template <typename T>
void WriteRowVectorFile(const std::filesystem::path& path, const RowVector<T>& vector) {
  std::ofstream out(path, std::ios::binary);
  assert(out.good());
  out.write(reinterpret_cast<const char*>(&vector[0]), sizeof(T) * vector.Size());
  assert(out.good());
}

template <typename T>
void WriteAffineStem(const std::filesystem::path& stem, const Matrix<T>& weights, const RowVector<T>& bias) {
  WriteMatrixFile(std::filesystem::path(stem.string() + "_w.bin"), weights);
  WriteRowVectorFile(std::filesystem::path(stem.string() + "_b.bin"), bias);
}

void ExpectNear(float actual, float expected, float tolerance) {
  assert(std::fabs(actual - expected) <= tolerance);
}

float GeluScalar(float x) {
  const float sqrt_two_over_pi = std::sqrt(2.0f / std::numbers::pi_v<float>);
  return 0.5f * x * (1.0f + std::tanh(sqrt_two_over_pi * (x + 0.044715f * x * x * x)));
}

Matrix<float> ManualLayerNorm(Matrix<float> x, const RowVector<float>& g, const RowVector<float>& b) {
  constexpr float eps = 1e-5f;
  for (int i = 0; i < x.Rows(); ++i) {
    float sum = 0.0f;
    for (int j = 0; j < x.Columns(); ++j)
      sum += x(i, j);
    const float mean = sum / x.Columns();

    float sumsqr = 0.0f;
    for (int j = 0; j < x.Columns(); ++j) {
      const float diff = x(i, j) - mean;
      sumsqr += diff * diff;
    }
    const float variance = sumsqr / x.Columns();
    const float scale = 1.0f / std::sqrt(variance + eps);

    for (int j = 0; j < x.Columns(); ++j)
      x(i, j) = (x(i, j) - mean) * scale * g[j] + b[j];
  }
  return x;
}

Matrix<float> ManualAffine(const Matrix<float>& x, const Matrix<float>& weights, const RowVector<float>& bias) {
  Matrix<float> out{x.Rows(), weights.Columns()};
  for (int i = 0; i < x.Rows(); ++i) {
    for (int j = 0; j < weights.Columns(); ++j) {
      float sum = 0.0f;
      for (int k = 0; k < x.Columns(); ++k)
        sum += x(i, k) * weights(k, j);
      out(i, j) = sum + bias[j];
    }
  }
  return out;
}

Matrix<float> ManualMlp(const Matrix<float>& x, const Affine<float>& c_fc, const Affine<float>& c_proj) {
  auto hidden = ManualAffine(x, c_fc.weights, c_fc.bias);
  for (int i = 0; i < hidden.Rows(); ++i)
    for (int j = 0; j < hidden.Columns(); ++j)
      hidden(i, j) = GeluScalar(hidden(i, j));
  return ManualAffine(hidden, c_proj.weights, c_proj.bias);
}

Matrix<float> ManualSingleHeadCausalAttention(const Matrix<float>& x) {
  Matrix<float> out{x.Rows(), x.Columns()};
  const float scale = 1.0f / std::sqrt(static_cast<float>(x.Columns()));
  for (int i = 0; i < x.Rows(); ++i) {
    std::vector<float> scores(i + 1);
    float max_score = -std::numeric_limits<float>::infinity();
    for (int j = 0; j <= i; ++j) {
      float dot = 0.0f;
      for (int k = 0; k < x.Columns(); ++k)
        dot += x(i, k) * x(j, k);
      scores[j] = dot * scale;
      max_score = std::max(max_score, scores[j]);
    }
    float denom = 0.0f;
    for (float& score : scores) {
      score = std::exp(score - max_score);
      denom += score;
    }
    for (float& score : scores)
      score /= denom;

    for (int col = 0; col < x.Columns(); ++col) {
      float value = 0.0f;
      for (int j = 0; j <= i; ++j)
        value += scores[j] * x(j, col);
      out(i, col) = value;
    }
  }
  return out;
}

std::filesystem::path TestTempDir() {
  const auto dir = std::filesystem::temp_directory_path() / "learngpt_infergpt_tests";
  std::filesystem::create_directories(dir);
  return dir;
}

void TestAffine() {
  Affine<float> affine(2, 3);
  affine.weights = MakeMatrix<float>(2, 3, {1, 2, 3, 4, 5, 6});
  FillRowVector(&affine.bias, {0.5f, -0.5f, 1.0f});

  const auto input = MakeMatrix<float>(2, 2, {1, 2, 3, 4});
  const auto actual = affine(input);
  const auto expected = ManualAffine(input, affine.weights, affine.bias);
  for (int i = 0; i < actual.Rows(); ++i)
    for (int j = 0; j < actual.Columns(); ++j)
      ExpectNear(actual(i, j), expected(i, j), 1e-6f);
}

void TestLayerNorm() {
  LayerNorm<float> layer_norm(3);
  FillRowVector(&layer_norm.g, {1.5f, 0.5f, 2.0f});
  FillRowVector(&layer_norm.b, {0.25f, -0.25f, 0.75f});

  const auto input = MakeMatrix<float>(2, 3, {1, 2, 3, 4, 2, 0});
  const auto actual = layer_norm(input);
  const auto expected = ManualLayerNorm(input, layer_norm.g, layer_norm.b);
  for (int i = 0; i < actual.Rows(); ++i)
    for (int j = 0; j < actual.Columns(); ++j)
      ExpectNear(actual(i, j), expected(i, j), 1e-5f);
}

void TestMultiHeadAttention() {
  const auto dir = TestTempDir();
  const auto stem = dir / "mha_test";

  const auto c_attn_weights = MakeMatrix<float>(2, 6, {
      1, 0, 1, 0, 1, 0,
      0, 1, 0, 1, 0, 1,
  });
  RowVector<float> c_attn_bias{6};
  FillRowVector(&c_attn_bias, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  const auto c_proj_weights = MakeMatrix<float>(2, 2, {1, 0, 0, 1});
  RowVector<float> c_proj_bias{2};
  FillRowVector(&c_proj_bias, {0.0f, 0.0f});
  WriteAffineStem(stem.string() + std::string{"_c_attn"}, c_attn_weights, c_attn_bias);
  WriteAffineStem(stem.string() + std::string{"_c_proj"}, c_proj_weights, c_proj_bias);

  MultiHeadAttention<float> attention(2, 1);
  attention.Load(stem.string());

  const auto input = MakeMatrix<float>(2, 2, {1, 0, 0, 1});
  const auto actual = attention(input);
  const auto expected = ManualSingleHeadCausalAttention(input);
  for (int i = 0; i < actual.Rows(); ++i)
    for (int j = 0; j < actual.Columns(); ++j)
      ExpectNear(actual(i, j), expected(i, j), 1e-5f);
}

void TestMultiLayerPerceptron() {
  MultiLayerPerceptron<float> mlp(2);
  mlp.c_fc.weights = MakeMatrix<float>(2, 8, {
      1, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0,
  });
  FillRowVector(&mlp.c_fc.bias, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  mlp.c_proj.weights = MakeMatrix<float>(8, 2, {
      1, 0,
      0, 1,
      0, 0,
      0, 0,
      0, 0,
      0, 0,
      0, 0,
      0, 0,
  });
  FillRowVector(&mlp.c_proj.bias, {0.0f, 0.0f});

  const auto input = MakeMatrix<float>(2, 2, {-1, 2, 0.5f, -0.25f});
  const auto actual = mlp(input);
  const auto expected = ManualMlp(input, mlp.c_fc, mlp.c_proj);
  for (int i = 0; i < actual.Rows(); ++i)
    for (int j = 0; j < actual.Columns(); ++j)
      ExpectNear(actual(i, j), expected(i, j), 1e-5f);
}

void TestTransformer() {
  const auto dir = TestTempDir();
  const auto stem = dir / "transformer_attn";

  const auto zero_qkv = MakeMatrix<float>(2, 6, {
      0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0,
  });
  RowVector<float> zero_qkv_bias{6};
  FillRowVector(&zero_qkv_bias, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  const auto zero_proj = MakeMatrix<float>(2, 2, {0, 0, 0, 0});
  RowVector<float> zero_proj_bias{2};
  FillRowVector(&zero_proj_bias, {0.0f, 0.0f});
  WriteAffineStem(stem.string() + std::string{"_c_attn"}, zero_qkv, zero_qkv_bias);
  WriteAffineStem(stem.string() + std::string{"_c_proj"}, zero_proj, zero_proj_bias);

  Transformer<float> transformer(2, 1);
  transformer.attention.Load(stem.string());
  FillRowVector(&transformer.ln_1.g, {1.0f, 1.0f});
  FillRowVector(&transformer.ln_1.b, {0.0f, 0.0f});
  FillRowVector(&transformer.ln_2.g, {1.0f, 1.0f});
  FillRowVector(&transformer.ln_2.b, {0.0f, 0.0f});
  transformer.mlp.c_fc.weights = MakeMatrix<float>(2, 8, {
      1, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0,
  });
  FillRowVector(&transformer.mlp.c_fc.bias, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  transformer.mlp.c_proj.weights = MakeMatrix<float>(8, 2, {
      1, 0,
      0, 1,
      0, 0,
      0, 0,
      0, 0,
      0, 0,
      0, 0,
      0, 0,
  });
  FillRowVector(&transformer.mlp.c_proj.bias, {0.0f, 0.0f});

  const auto input = MakeMatrix<float>(2, 2, {1, 2, 3, 4});
  const auto actual = transformer(input);

  const auto normed = ManualLayerNorm(input, transformer.ln_2.g, transformer.ln_2.b);
  const auto mlp_out = ManualMlp(normed, transformer.mlp.c_fc, transformer.mlp.c_proj);
  auto expected = input;
  expected += mlp_out;

  for (int i = 0; i < actual.Rows(); ++i)
    for (int j = 0; j < actual.Columns(); ++j)
      ExpectNear(actual(i, j), expected(i, j), 1e-5f);
}

}  // namespace

int main() {
  TestAffine();
  TestLayerNorm();
  TestMultiHeadAttention();
  TestMultiLayerPerceptron();
  TestTransformer();
}