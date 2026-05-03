#pragma once

#include <cassert>
#include <tuple>
#include <utility>

#include "matrix.h"

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
  using R = typename std::remove_cvref_t<M>;
  Matrix<typename R::Scalar> result{m.Rows(), m.Columns()};
  TransformImpl(m, &result, std::forward<F>(fn));
  return result;
}

template <IsMatrix Src1, IsMatrix Src2, IsMatrix Dst, typename F>
void BinaryTransformImpl(const Src1& src1, const Src2& src2, Dst* dst, F&& fn) {
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

template <typename T, IsMatrix R, typename F>
Matrix<T> BinaryTransform(Matrix<T> m, const R& rhs, F&& fn) {
  BinaryTransformImpl(m, rhs, &m, std::forward<F>(fn));
  return m;
}

template <IsMatrix M1, IsMatrix M2, typename F>
auto BinaryTransform(const M1& lhs, const M2& rhs, F&& fn) {
  assert(lhs.Rows() == rhs.Rows());
  assert(lhs.Columns() == rhs.Columns());
  using T = std::remove_cvref_t<decltype(std::invoke(std::forward<F>(fn), lhs(0, 0), rhs(0, 0)))>;
  Matrix<T> result{lhs.Rows(), lhs.Columns()};
  BinaryTransformImpl(lhs, rhs, &result, std::forward<F>(fn));
  return result;
}

template <IsVector L, IsVector R>
auto Dot(const L& lhs, const R& rhs, int count = -1) {
  assert(lhs.Size() == rhs.Size());
  using T = std::decay_t<decltype(std::declval<typename L::Scalar>() * std::declval<typename R::Scalar>())>;
  if (count < 0) count = lhs.Size();

  T sum = T{0};
  for (int i = 0; i < count; ++i)
    sum += lhs[i] * rhs[i];
  return sum;
}

template <IsVector V>
auto MeanAndVariance(const V& v) {
  using T = typename std::remove_cvref_t<V>::Scalar;
  const T scale = T{1} / v.Size();

  T sum = T{0};
  for (int i = 0; i < v.Size(); ++i)
    sum += v[i];
  const T mean = sum * scale;

  T sumsqr = T{0};
  for (int i = 0; i < v.Size(); ++i) {
    T diff = v[i] - mean;
    sumsqr += diff * diff;
  }
  const T var = sumsqr * scale;

  return std::make_pair(mean, var);
}

template <IsMatrix L, IsMatrix R>
auto operator*(const L& lhs, const R& rhs) {
  using T = decltype(std::declval<typename L::Scalar>() * std::declval<typename R::Scalar>());
  assert(lhs.Columns() == rhs.Rows());

  Matrix<T> out(lhs.Rows(), rhs.Columns());
  for (int i = 0; i < lhs.Rows(); ++i) {
    const T* row = lhs[i];
    for (int j = 0; j < rhs.Columns(); ++j) {
      T sum = T{0};
      for (int k = 0; k < lhs.Columns(); ++k)
        sum += row[k] * rhs(k, j);
      out(i, j) = sum;
    }
  }
  return out;
}

template <IsMatrix L, IsMatrix R>
L& operator+=(L& lhs, const R& rhs) {
  using T = typename L::Scalar;
  assert(lhs.Rows() == rhs.Rows());
  assert(lhs.Columns() == rhs.Columns());
  for (int i = 0; i < lhs.Rows(); ++i) {
    T* pl = lhs[i];
    const T* pr = rhs[i];
    for (int j = 0; j < lhs.Columns(); ++j)
      pl[j] += pr[j];
  }
  return lhs;
}

template <IsMatrix L, IsMatrix R>
auto operator+(const L& lhs, const R& rhs) {
  using T = decltype(std::declval<typename L::Scalar>() + std::declval<typename R::Scalar>());
  assert(lhs.Rows() == rhs.Rows());
  assert(lhs.Columns() == rhs.Columns());
  Matrix<T> out{lhs.Rows(), lhs.Columns()};
  for (int i = 0; i < lhs.Rows(); ++i) {
    T* pd = out[i];
    const T* pl = lhs[i];
    const T* pr = rhs[i];
    for (int j = 0; j < lhs.Columns(); ++j)
      pd[j] = pl[j] + pr[j];
  }
  return out;
}