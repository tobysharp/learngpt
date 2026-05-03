#pragma once

#include <cassert>
#include <concepts>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

template <typename M>
concept IsMatrix = requires(M m, int i, int j) {
  typename std::remove_cvref_t<M>::Scalar;
  { m.Rows() } -> std::convertible_to<int>;
  { m.Columns() } -> std::convertible_to<int>;
  { m[i] };
  { m(i, j) };
};

template <IsMatrix M>
class SubMatrixView {
 public:
  using Scalar = typename std::remove_cvref_t<M>::Scalar;

  SubMatrixView(M& matrix, int row_begin, int col_begin, int rows, int cols) :
     matrix_(matrix), row_index_(row_begin, row_begin + rows), col_index_(col_begin, col_begin + cols) {}
  
  int Rows() const { return row_index_.second - row_index_.first; }
  int Columns() const { return col_index_.second - col_index_.first; }
  decltype(auto) operator[](int row) const { return &matrix_.get()(row_index_.first + row, col_index_.first); }
  decltype(auto) operator()(int row, int col) const { return matrix_.get()(row_index_.first + row, col_index_.first + col); }

  M& Base() { return matrix_.get(); }
  const M& Base() const { return matrix_.get(); }
  int RowBegin() const { return row_index_.first; }
  int ColBegin() const { return col_index_.first; }

  template <IsMatrix Rhs>
    requires (!std::is_const_v<std::remove_reference_t<M>>)
  SubMatrixView& operator=(const Rhs& rhs) {
    using R = std::remove_cvref_t<Rhs>;
    static_assert(std::is_convertible_v<typename R::Scalar, Scalar>);
    assert(rhs.Rows() == Rows());
    assert(rhs.Columns() == Columns());

    for (int i = 0; i < Rows(); ++i)
    {
      const auto* src = rhs[i];
      auto* dst = (*this)[i];
      for (int j = 0; j < Columns(); ++j)
        dst[j] = src[j];
    }
    return *this;
  }

 private:
  std::reference_wrapper<M> matrix_; 
  std::pair<int, int> row_index_;
  std::pair<int, int> col_index_;
};

template <IsMatrix M>
auto Block(M& m, int row, int col, int rows, int cols) {
  return SubMatrixView<M>(m, row, col, rows, cols);
}

template <IsMatrix M>
auto Block(SubMatrixView<M>& m, int row, int col, int rows, int cols) {
  return SubMatrixView<M>{m.Base(), m.RowBegin() + row, m.ColBegin() + col, rows, cols}; 
}

template <IsMatrix M>
auto Block(const SubMatrixView<M>& m, int row, int col, int rows, int cols) {
  return SubMatrixView<const M>{m.Base(), m.RowBegin() + row, m.ColBegin() + col, rows, cols}; 
}

template <typename T>
class Matrix {
 public:
  using Scalar = T;

  Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows * cols) {}
  Matrix(int rows, int cols, std::vector<T>&& data) : rows_(rows), cols_(cols), data_(std::move(data)) {}

  int Rows() const { return rows_; }
  int Columns() const { return cols_; }
  
  T* RowData(int row = 0) { return data_.data() + row * cols_; }
  const T* RowData(int row = 0) const { return data_.data() + row * cols_; }

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

 protected:
  int rows_, cols_;
  std::vector<T> data_;
};

template <IsMatrix M>
auto Transpose(const M& m) {
  using R = typename std::remove_cvref_t<M>;
  Matrix<typename R::Scalar> result{m.Columns(), m.Rows()};
  for (int i = 0; i < m.Rows(); ++i) {
    const auto* src = m[i];
    for (int j = 0; j < m.Columns(); ++j)
      result(j, i) = src[j];
  }
  return result;
}

template <typename V>
concept IsVector = requires(V v, int i) {
  typename std::remove_cvref_t<V>::Scalar;
  { v.Size() } -> std::convertible_to<int>;
  { v[i] };
};

// A row vector.
template <typename T>
class RowVector : public Matrix<T> {
 public:
  explicit RowVector(int cols) : Matrix<T>(1, cols) {}
  RowVector(std::vector<T>&& data) : Matrix<T>(1, std::ssize(data), std::move(data)) {}
  RowVector(const RowVector&) = default;
  RowVector(RowVector&&) = default;

  int Size() const { return this->Columns(); }

  T& operator[](int index) { return this->data_[index]; }
  const T& operator[](int index) const { return this->data_[index]; }

  operator std::span<T>() { return this->data_; }
  operator std::span<const T>() const { return this->data_; }
};

template <IsMatrix M>
class RowView {
 public:
  using Scalar = typename std::remove_cvref_t<M>::Scalar;
  using Reference = decltype(std::declval<M&>()(0, 0));
  using Pointer = decltype(std::declval<M&>()[0]);

  RowView(M& matrix, int row) :
     matrix_(matrix), data_(matrix[row]) {}
  
  int Rows() const { return 1; }
  int Columns() const { return matrix_.get().Columns(); }
  int Size() const { return Columns(); }

  Reference operator[](int col) const { return data_[col]; }

 private:
  std::reference_wrapper<M> matrix_; 
  Pointer data_;
};

template <IsMatrix M>
auto Row(M& m, int row) {
  assert(row < m.Rows());
  return RowView<M>{m, row};
}

template <IsMatrix M>
class ColumnView {
 public:
  using Scalar = typename  std::remove_cvref_t<M>::Scalar;
  using Reference = decltype(std::declval<M&>()(0, 0));
  using Pointer = decltype(std::declval<M&>()[0]);

  ColumnView(M& matrix, int col) :
     matrix_(matrix), column_(col) {}
  
  int Rows() const { return matrix_.get().Rows(); }
  int Columns() const { return 1; }
  int Size() const { return Rows(); }

  Reference operator[](int row) const { return matrix_.get()(row, column_); }

 private:
  std::reference_wrapper<M> matrix_; 
  int column_;
};

template <IsMatrix M>
auto Column(M& m, int column) {
  assert(column < m.Columns());
  return ColumnView<M>{m, column};
}

template <IsVector V>
class RowBroadcastView {
 public:
  using Scalar = typename std::remove_cvref_t<V>::Scalar;
  using Reference = decltype(std::declval<V&>()[0]);
  using Pointer = decltype(&std::declval<V&>()[0]);

  RowBroadcastView(V& vector, int rows) :
     vector_(vector), rows_(rows) {}
  
  int Rows() const { return rows_; }
  int Columns() const { return vector_.get().Size(); }
  
  Pointer operator[](int row) const { return &vector_.get()[0]; }
  Reference operator()(int row, int col) const { return vector_.get()[col]; }

 private:
  std::reference_wrapper<V> vector_; 
  const int rows_;
};

template <IsVector V>
auto BroadcastToRows(V& v, int rows) {
  return RowBroadcastView<V>{v, rows};
}