#pragma once

#include <cassert>
#include <concepts>
#include <exception>
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
  int RowBegin() const { return row_index_.first; }
  int ColBegin() const { return col_index_.first; }

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

template <typename T = float>
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
  Matrix<typename std::remove_cvref_t<M>::Scalar> result{m.Columns(), m.Rows()};
  for (int i = 0; i < m.Rows(); ++i) {
    const auto* src = m[i];
    for (int j = 0; j < m.Columns(); ++j)
      result(j, i) = src[j];
  }
  return result;
}

template <IsMatrix Src, IsMatrix Dst, typename F>
void TransformImpl(const Src& src, Dst* dst, F&& fn) {
  assert(src.Rows() == dst->Rows());
  assert(src.Columns() == dst->Columns());
  for (int i = 0; i < src.Rows(); ++i) {
    const auto* s = src[i];
    auto* d = (*dst)[i];
    for (int j = 0; j < src.Columns(); ++j)
      d[j] = std::invoke(fn, s[j]);
  }
}

template <typename T, typename F>
Matrix<T> Transform(Matrix<T> m, F&& fn) {
  TransformImpl(m, &m, std::forward<F>(fn));
  return m;
}

template <IsMatrix M, typename F>
auto Transform(const M& m, F&& fn) {
  using T = typename std::remove_cvref_t<M>::Scalar;
  Matrix<T> result{m.Rows(), m.Columns()};
  TransformImpl(m, &result, std::forward<F>(fn));
  return result;
}

// A row vector.
template <typename T>
class Vector : public Matrix<T> {
 public:
  explicit Vector(int cols) : Matrix<T>(1, cols) {}
  Vector(std::vector<T>&& data) : Matrix<T>(1, std::ssize(data), std::move(data)) {}
  Vector(const Vector&) = default;
  Vector(Vector&&) = default;

  int Size() const { return this->Columns(); }

  T& operator[](int index) { return this->RowData()[index]; }
  const T& operator[](int index) const { return this->RowData()[index]; }

  operator std::span<T>() { return this->data_; }
  operator std::span<const T>() const { return this->data_; }
};
