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
    const auto* s = src.RowData(i);
    auto* d = dst->RowData(i);
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

template <IsVector L, IsVector R>
auto Dot(const L& lhs, const R& rhs, int count = -1) {
  assert(lhs.Size() == rhs.Size());
  using T = std::decay_t<decltype(std::declval<typename L::Scalar>() * std::declval<typename R::Scalar>())>;
  if (count < 0) count = lhs.Size();

  T sum = T{0};
  for (int i = 0; i < count; ++i)
    sum += lhs(i) * rhs(i);
  return sum;
}

template <IsVector V>
auto MeanAndVariance(const V& v) {
  using T = typename std::remove_cvref_t<V>::Scalar;
  const T scale = T{1} / v.Size();

  T sum = T{0};
  for (int i = 0; i < v.Size(); ++i)
    sum += v(i);
  const T mean = sum * scale;

  T sumsqr = T{0};
  for (int i = 0; i < v.Size(); ++i) {
    T diff = v(i) - mean;
    sumsqr += diff * diff;
  }
  const T var = sumsqr * scale;

  return std::make_pair(mean, var);
}

template <IsMatrix X, IsMatrix Y>
auto MatMul_XYT(const X& lhs, const Y& rhs) {
  using T = decltype(std::declval<typename X::Scalar>() * std::declval<typename Y::Scalar>());
  assert(lhs.Columns() == rhs.Columns());

  const int lrows = lhs.Rows();
  const int lcols = lhs.Columns();
  const int rrows = rhs.Rows();
  Matrix<T> out(lrows, rrows);
  for (int i = 0; i < lrows; ++i) {
    const T* pl = lhs.RowData(i);
    T* pout = out.RowData(i);
    for (int j = 0; j < rrows; ++j) {
      T sum = T{0};
      const T* pr = rhs.RowData(j);
      for (int k = 0; k < lcols; ++k)
        sum += pl[k] * pr[k];
      pout[j] = sum;
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
    T* pl = lhs.RowData(i);
    const T* pr = rhs.RowData(i);
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
    T* pd = out.RowData(i);
    const T* pl = lhs.RowData(i);
    const T* pr = rhs.RowData(i);
    for (int j = 0; j < lhs.Columns(); ++j)
      pd[j] = pl[j] + pr[j];
  }
  return out;
}

template <IsVector V>
int ArgMax(const V& v) {
  using T = typename V::Scalar;
  assert(v.Size() > 0);
  T max = v(0);
  int pos = 0;
  for (int i = 1; i < v.Size(); ++i) {
    if (v(i) > max) {
      pos = i;
      max = v(i);
    }
  }
  return pos;
}