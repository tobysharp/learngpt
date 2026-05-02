#pragma once

#include <cassert>
#include <tuple>
#include <utility>

#include "matrix.h"

template <IsMatrix L, IsMatrix R>
auto RowDotRow(const L& lhs, int lrow, const R& rhs, int rrow) {
  assert(lhs.Columns() == rhs.Columns());
  using T = std::decay_t<decltype(std::declval<typename L::Scalar>() * std::declval<typename R::Scalar>())>;
  T sum = T{0};
  const auto* pl = lhs[lrow];
  const auto* pr = rhs[rrow];
  for (int i = 0; i < lhs.Columns(); ++i)
    sum += pl[i] * pr[i];
  return sum;
}

template <IsMatrix L, IsMatrix R>
auto RowDotColumn(const L& lhs, int lrow, const R& rhs, int rcol, int count = -1) {
  assert(lhs.Columns() == rhs.Rows());
  assert(count <= lhs.Columns());
  using T = std::decay_t<decltype(std::declval<typename L::Scalar>() * std::declval<typename R::Scalar>())>;
  if (count < 0) count = lhs.Columns();
  T sum = T{0};
  const auto* pl = lhs[lrow];
  for (int i = 0; i < count; ++i)
    sum += pl[i] * rhs(i, rcol);
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