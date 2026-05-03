#include <cassert>
#include <type_traits>

#include "algebra.h"
#include "matrix.h"

namespace {

using TestMatrix = Matrix<float>;

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

  const auto row0 = Row(matrix, 0);
  const auto row1 = Row(matrix, 1);
  const auto col2 = Column(matrix, 2);

  assert(Dot(row0, row1) == 32.0f);
  assert(Dot(row0, col2) == 42.0f);
  assert(Dot(row1, col2, 2) == 42.0f);

  RowVector<float> weights{3};
  weights[0] = 0.5f;
  weights[1] = 1.5f;
  weights[2] = -1.0f;
  assert(Dot(row0, weights) == 0.5f * 1.0f + 1.5f * 2.0f - 3.0f);
}

void TestTransformMutatesMovedMatrixAndPreservesAxes() {
  TestMatrix matrix{2, 3};
  Fill(&matrix);

  const auto doubled = Transform(matrix, [](float value) { return value * 2.0f; });
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