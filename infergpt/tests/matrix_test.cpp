#include <cassert>
#include <type_traits>

#include "matrix.h"

namespace {

struct Rows {};
struct Cols {};
struct InnerRows {};
struct InnerCols {};

using MutableMatrix = Matrix<int, Rows, Cols>;
using ConstMatrix = const MutableMatrix;

static_assert(IsMatrix<MutableMatrix>);
static_assert(IsVector<RowVector<int, Cols>>);
static_assert(std::is_same_v<decltype(std::declval<RowView<MutableMatrix, Cols>>()[0]), int&>);
static_assert(std::is_same_v<decltype(std::declval<RowView<ConstMatrix, Cols>>()[0]), const int&>);
static_assert(std::is_same_v<decltype(std::declval<ColumnView<MutableMatrix, Rows>>()[0]), int&>);
static_assert(std::is_same_v<decltype(std::declval<ColumnView<ConstMatrix, Rows>>()[0]), const int&>);
static_assert(!std::is_assignable_v<SubMatrixView<const MutableMatrix, Rows, Cols>&, const MutableMatrix&>);

void FillSequential(MutableMatrix* matrix) {
  int value = 0;
  for (int row = 0; row < matrix->Rows(); ++row) {
    for (int col = 0; col < matrix->Columns(); ++col)
      (*matrix)(row, col) = value++;
  }
}

void TestRowAndColumnViewsPreserveConstnessAndMutability() {
  MutableMatrix matrix{3, 4};
  FillSequential(&matrix);

  auto row = Row(matrix, 1);
  static_assert(std::is_same_v<decltype(row[0]), int&>);
  row[2] = 99;
  assert(matrix(1, 2) == 99);

  auto column = Column(matrix, 3);
  static_assert(std::is_same_v<decltype(column[0]), int&>);
  column[2] = -7;
  assert(matrix(2, 3) == -7);

  const MutableMatrix& const_matrix = matrix;
  auto const_row = Row(const_matrix, 0);
  auto const_column = Column(const_matrix, 1);
  static_assert(std::is_same_v<decltype(const_row[0]), const int&>);
  static_assert(std::is_same_v<decltype(const_column[0]), const int&>);
  assert(const_row[3] == matrix(0, 3));
  assert(const_column[2] == matrix(2, 1));
}

void TestBlockViewMutationAndNestedBlocks() {
  MutableMatrix matrix{4, 5};
  FillSequential(&matrix);

  auto outer = Block<Rows, Cols>(matrix, 1, 1, 2, 3);
  outer(0, 0) = 111;
  outer(1, 2) = 222;
  assert(matrix(1, 1) == 111);
  assert(matrix(2, 3) == 222);

  Matrix<int, Rows, Cols> replacement{2, 3};
  FillSequential(&replacement);
  outer = replacement;
  assert(matrix(1, 1) == 0);
  assert(matrix(2, 3) == 5);

  auto nested = Block<InnerRows, InnerCols>(outer, 0, 1, 2, 2);
  nested(1, 1) = 333;
  assert(matrix(2, 3) == 333);

  const auto& const_outer = outer;
  auto const_nested = Block<InnerRows, InnerCols>(const_outer, 0, 0, 1, 2);
  assert(const_nested(0, 1) == matrix(1, 2));
}

void TestTransposeAndRowVectorStorage() {
  MutableMatrix matrix{2, 3};
  FillSequential(&matrix);

  const auto transposed = Transpose(matrix);
  static_assert(std::is_same_v<typename decltype(transposed)::RowAxis, Cols>);
  static_assert(std::is_same_v<typename decltype(transposed)::ColAxis, Rows>);
  assert(transposed.Rows() == 3);
  assert(transposed.Columns() == 2);
  assert(transposed(2, 1) == matrix(1, 2));

  RowVector<int, Cols> vector{4};
  for (int i = 0; i < vector.Size(); ++i)
    vector[i] = i + 10;
  assert(vector.Rows() == 1);
  assert(vector.Columns() == 4);
  assert(vector[3] == 13);
}

}  // namespace

int main() {
  TestRowAndColumnViewsPreserveConstnessAndMutability();
  TestBlockViewMutationAndNestedBlocks();
  TestTransposeAndRowVectorStorage();
}