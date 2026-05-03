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

  const auto sum = lhs + rhs;
  assert(sum(0, 0) == 2.0f);
  assert(sum(1, 2) == 12.0f);

  const auto scaled_rhs = Transform(rhs, [](float value) { return value * -0.5f; });
  const auto scaled = lhs + scaled_rhs;
  assert(scaled(0, 1) == 1.0f);
  assert(scaled(1, 2) == 3.0f);

  const auto [mean, variance] = MeanAndVariance(Row(rhs, 1));
  assert(mean == 5.0f);
  assert(variance == 2.0f / 3.0f);
}

void TestMatrixArithmeticOperators() {
  TestMatrix lhs{2, 3};
  TestMatrix rhs{3, 2};
  Fill(&lhs);
  Fill(&rhs);

  const auto product = MatMul_XYT(lhs, Transpose(rhs));
  assert(product.Rows() == 2);
  assert(product.Columns() == 2);
  assert(product(0, 0) == 22.0f);
  assert(product(0, 1) == 28.0f);
  assert(product(1, 0) == 49.0f);
  assert(product(1, 1) == 64.0f);

  TestMatrix add_lhs{2, 2};
  TestMatrix add_rhs{2, 2};
  add_lhs(0, 0) = 1.0f; add_lhs(0, 1) = 2.0f;
  add_lhs(1, 0) = 3.0f; add_lhs(1, 1) = 4.0f;
  add_rhs(0, 0) = 10.0f; add_rhs(0, 1) = 20.0f;
  add_rhs(1, 0) = 30.0f; add_rhs(1, 1) = 40.0f;

  const auto sum = add_lhs + add_rhs;
  assert(sum(0, 0) == 11.0f);
  assert(sum(1, 1) == 44.0f);

  add_lhs += add_rhs;
  assert(add_lhs(0, 1) == 22.0f);
  assert(add_lhs(1, 0) == 33.0f);
}

void TestBroadcastAddition() {
  TestMatrix matrix{2, 2};
  matrix(0, 0) = 1.0f; matrix(0, 1) = 2.0f;
  matrix(1, 0) = 3.0f; matrix(1, 1) = 4.0f;

  RowVector<float> bias{2};
  bias[0] = 1.0f;
  bias[1] = 2.0f;

  const auto sum = matrix + BroadcastToRows(bias, 2);
  assert(sum(0, 0) == 2.0f);
  assert(sum(0, 1) == 4.0f);
  assert(sum(1, 0) == 4.0f);
  assert(sum(1, 1) == 6.0f);
}

}  // namespace

int main() {
  TestDotAcrossVectorViews();
  TestTransformMutatesMovedMatrixAndPreservesAxes();
  TestBinaryTransformAndRowStatistics();
  TestMatrixArithmeticOperators();
  TestBroadcastAddition();
}