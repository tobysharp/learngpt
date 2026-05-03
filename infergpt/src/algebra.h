#pragma once

#include <cassert>
#include <tuple>
#include <utility>

#include "matrix.h"

template <IsMatrix Src, IsMatrix Dst, typename F>
void TransformImpl(const Src& src, Dst* dst, F&& fn) {
  using L = typename std::remove_cvref_t<Src>;
  using R = typename std::remove_cvref_t<Dst>;
  static_assert(std::is_same_v<typename L::RowAxis, typename R::RowAxis>);
  static_assert(std::is_same_v<typename L::ColAxis, typename R::ColAxis>);
  assert(src.Rows() == dst->Rows());
  assert(src.Columns() == dst->Columns());

  for (int i = 0; i < src.Rows(); ++i) {
    const auto* s = src[i];
    auto* d = (*dst)[i];
    for (int j = 0; j < src.Columns(); ++j)
      d[j] = std::invoke(fn, s[j]);
  }
}

template <typename T, typename RowAxis, typename ColAxis, typename F>
Matrix<T, RowAxis, ColAxis> Transform(Matrix<T, RowAxis, ColAxis> m, F&& fn) {
  TransformImpl(m, &m, std::forward<F>(fn));
  return m;
}

template <IsMatrix M, typename F>
auto Transform(const M& m, F&& fn) {
  using R = typename std::remove_cvref_t<M>;
  Matrix<typename R::Scalar, typename R::RowAxis, typename R::ColAxis> result{m.Rows(), m.Columns()};
  TransformImpl(m, &result, std::forward<F>(fn));
  return result;
}

template <IsMatrix Src1, IsMatrix Src2, IsMatrix Dst, typename F>
void BinaryTransformImpl(const Src1& src1, const Src2& src2, Dst* dst, F&& fn) {
  using L1 = typename std::remove_cvref_t<Src1>;
  using L2 = typename std::remove_cvref_t<Src2>;
  using R = typename std::remove_cvref_t<Dst>;
  static_assert(std::is_same_v<typename L1::RowAxis, typename L2::RowAxis>);
  static_assert(std::is_same_v<typename L1::ColAxis, typename L2::ColAxis>);
  static_assert(std::is_same_v<typename L2::RowAxis, typename R::RowAxis>);
  static_assert(std::is_same_v<typename L2::ColAxis, typename R::ColAxis>);
  assert(src1.Rows() == dst->Rows());
  assert(src1.Rows() == src2.Rows());
  assert(src1.Columns() == dst->Columns());
  assert(src1.Columns() == src2.Columns());

  for (int i = 0; i < src1.Rows(); ++i) {
    const auto* s1 = src1[i];
    const auto* s2 = src2[i];
    auto* d = (*dst)[i];
    for (int j = 0; j < src1.Columns(); ++j)
      d[j] = std::invoke(fn, s1[j], s2[j]);
  }
}

template <typename T, typename RowAxis, typename ColAxis, IsMatrix R, typename F>
Matrix<T, RowAxis, ColAxis> BinaryTransform(Matrix<T, RowAxis, ColAxis> m, const R& rhs, F&& fn) {
  BinaryTransformImpl(m, rhs, &m, std::forward<F>(fn));
  return m;
}

template <IsMatrix M1, IsMatrix M2, typename F>
auto BinaryTransform(const M1& lhs, const M2& rhs, F&& fn) {
  using R1 = typename std::remove_cvref_t<M1>;
  using R2 = typename std::remove_cvref_t<M2>;
  static_assert(std::is_same_v<typename R1::RowAxis, typename R2::RowAxis>);
  static_assert(std::is_same_v<typename R1::ColAxis, typename R2::ColAxis>);
  assert(lhs.Rows() == rhs.Rows());
  assert(lhs.Columns() == rhs.Columns());
  using T = std::remove_cvref_t<decltype(std::invoke(std::forward<F>(fn), lhs(0, 0), rhs(0, 0)))>;
  Matrix<T, typename R1::RowAxis, typename R1::ColAxis> result{lhs.Rows(), lhs.Columns()};
  BinaryTransformImpl(lhs, rhs, &result, std::forward<F>(fn));
  return result;
}

template <IsVector L, IsVector R>
auto Dot(const L& lhs, const R& rhs, int count = -1) {
  using X = std::remove_cvref_t<L>;
  using Y = std::remove_cvref_t<R>;
  using LSize = typename X::Axis;
  using RSize = typename Y::Axis;
  static_assert(std::is_same_v<LSize, RSize>);
  assert(lhs.Size() == rhs.Size());
  using T = std::decay_t<decltype(std::declval<typename L::Scalar>() * std::declval<typename R::Scalar>())>;
  if (count < 0) count = lhs.Size();

  T sum = T{0};
  for (int i = 0; i < count; ++i)
    sum += lhs[i] * rhs[i];
  return sum;
}

template <IsMatrix M>
auto RowMeanAndVariance(const M& m, int row) {
  using T = typename std::remove_cvref_t<M>::Scalar;
  const T scale = T{1} / m.Columns();
  const T* src = m[row];

  T sum = T{0};
  for (int i = 0; i < m.Columns(); ++i)
    sum += src[i];
  const T mean = sum * scale;

  T sumsqr = T{0};
  for (int i = 0; i < m.Columns(); ++i) {
    T diff = src[i] - mean;
    sumsqr += diff * diff;
  }
  const T var = sumsqr * scale;

  return std::make_pair(mean, var);
}

template <IsMatrix L, IsMatrix R>
auto operator*(const L& lhs, const R& rhs) {
  // Template magic.
  using LM = std::remove_cvref_t<L>;
  using RM = std::remove_cvref_t<R>;
  using LRows = typename LM::RowAxis;
  using LCols = typename LM::ColAxis;
  using RRows = typename RM::RowAxis;
  using RCols = typename RM::ColAxis;
  using LScalar = typename LM::Scalar;
  using RScalar = typename RM::Scalar;
  using T = decltype(std::declval<LScalar>() * std::declval<RScalar>());

  // Compile-time shape validation.
  static_assert(std::is_same_v<LCols, RRows>);

  // Runtime size validation.
  assert(lhs.Columns() == rhs.Rows());

  Matrix<T, LRows, RCols> out(lhs.Rows(), rhs.Columns());
  // TODO
  return out;
}

template <IsMatrix L, IsMatrix R>
L& operator+=(L& lhs, const R& rhs) {
  // TODO
  return lhs;
}

template <IsMatrix L, IsMatrix R>
auto operator+(const L& lhs, const R& rhs) {
  using LM = std::remove_cvref_t<L>;
  using RM = std::remove_cvref_t<R>;
  using LRows = typename LM::RowAxis;
  using LCols = typename LM::ColAxis;
  using RRows = typename RM::RowAxis;
  using RCols = typename RM::ColAxis;
  using LScalar = typename LM::Scalar;
  using RScalar = typename RM::Scalar;  
  // TODO
  return lhs;
}