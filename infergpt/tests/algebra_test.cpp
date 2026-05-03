#include <cassert>
#include <type_traits>

#include "algebra.h"

namespace {

struct Rows {};
struct Cols {};
struct SquareAxis {};

using TestMatrix = Matrix<float, Rows, Cols>;
using SquareMatrix = Matrix<float, SquareAxis, SquareAxis>;

void Fill(TestMatrix* matrix) {
  float value = 1.0f;
  for (int row = 0; row < matrix->Rows(); ++row) {
    for (int col = 0; col < matrix->Columns(); ++col)
      (*matrix)(row, col) = value++;
  }
}

void TestDotAcrossVectorViews() {
  TestMatrix matrix{3, 3};
  Fill(&matrix);
  SquareMatrix square{3, 3};
  float value = 1.0f;
  for (int row = 0; row < square.Rows(); ++row) {
    for (int col = 0; col < square.Columns(); ++col)
      square(row, col) = value++;
  }

  const auto row0 = Row(matrix, 0);
  const auto row1 = Row(matrix, 1);
  const auto square_row0 = Row(square, 0);
  const auto square_col2 = Column(square, 2);

  assert(Dot(row0, row1) == 32.0f);
  assert(Dot(square_row0, square_col2) == 42.0f);
  assert(Dot(square_row0, square_col2, 2) == 15.0f);

  RowVector<float, Cols> weights{3};
  weights[0] = 0.5f;
  weights[1] = 1.5f;
  weights[2] = -1.0f;
  assert(Dot(row0, weights) == 0.5f * 1.0f + 1.5f * 2.0f - 3.0f);
}

void TestTransformMutatesMovedMatrixAndPreservesAxes() {
  TestMatrix matrix{2, 3};
  Fill(&matrix);

  const auto doubled = Transform(matrix, [](float value) { return value * 2.0f; });
  static_assert(std::is_same_v<typename decltype(doubled)::RowAxis, Rows>);
  static_assert(std::is_same_v<typename decltype(doubled)::ColAxis, Cols>);
  assert(doubled(1, 2) == 12.0f);
  assert(matrix(1, 2) == 6.0f);

  const auto shifted = Transform(std::move(matrix), [](float value) { return value + 1.0f; });
  assert(shifted(0, 0) == 2.0f);
  assert(shifted(1, 2) == 7.0f);
}

void TestBinaryTransformAndRowStatistics() {
  TestMatrix lhs{2, 3};
  TestMatrix rhs{2, 3};
  Fill(&lhs);
  Fill(&rhs);

  const auto sum = BinaryTransform(lhs, rhs, [](float left, float right) { return left + right; });
  assert(sum(0, 0) == 2.0f);
  assert(sum(1, 2) == 12.0f);

  const auto scaled = BinaryTransform(std::move(lhs), rhs, [](float left, float right) { return left - right * 0.5f; });
  assert(scaled(0, 1) == 1.0f);
  assert(scaled(1, 2) == 3.0f);

  const auto [mean, variance] = RowMeanAndVariance(rhs, 1);
  assert(mean == 5.0f);
  assert(variance == 2.0f / 3.0f);
}

}  // namespace

int main() {
  TestDotAcrossVectorViews();
  TestTransformMutatesMovedMatrixAndPreservesAxes();
  TestBinaryTransformAndRowStatistics();
}