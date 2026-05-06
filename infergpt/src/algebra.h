#pragma once

#include <cassert>
#include <tuple>
#include <utility>

#include "matrix.h"
#include "pfor.h"

template <IsMatrix Src, IsMatrix Dst, typename F>
  requires (!IsVector<Src>)
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

template <IsVector Src, IsVector Dst, typename F>
void TransformImpl(const Src& src, Dst* dst, F&& fn) {
  assert(src.Size() == dst->Size());

  for (int i = 0; i < src.Size(); ++i)
    (*dst)(i) = std::invoke(fn, src(i));
}

template <WritableTensor X, typename F>
X Transform(X x, F&& fn) {
  TransformImpl(x, &x, std::forward<F>(fn));
  return x;
}

template <IsTensor X, typename F> requires (!WritableTensor<X>)
ValueTensor<X> Transform(const X& x, F&& fn) {
  ValueTensor<X> result = AllocateShape(x);
  TransformImpl(x, &result, std::forward<F>(fn));
  return result;
}

template <IsVector L, IsVector R>
auto Dot(const L& lhs, const R& rhs) {
  assert(lhs.Size() == rhs.Size());
  using T = std::decay_t<decltype(std::declval<typename L::Scalar>() * std::declval<typename R::Scalar>())>;

  T sum = T{0};
  for (int i = 0; i < lhs.Size(); ++i)
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

template <typename T>
inline T XYT_Kernel(const T* pl, const T* pr, int lcols)
{
  T sum = T{0};
  for (int k = 0; k < lcols; ++k)
    sum += pl[k] * pr[k];
  return sum;
}

template <IsMatrix X, IsMatrix Y>
auto MatMul_XYT(const X& lhs, const Y& rhs) {
  using T = decltype(std::declval<typename X::Scalar>() * std::declval<typename Y::Scalar>());
  assert(lhs.Columns() == rhs.Columns());

  const int lrows = lhs.Rows();
  const int lcols = lhs.Columns();
  const int rrows = rhs.Rows();
  Matrix<T> out(lrows, rrows);

  constexpr int lrows_per_block = 16;
  constexpr int rrows_per_block = 32;
  const int lrow_blocks = (lrows + lrows_per_block - 1) / lrows_per_block;
  const int rrow_blocks = (rrows + rrows_per_block - 1) / rrows_per_block;
  const int total_blocks = lrow_blocks * rrow_blocks;
  
  ParallelFor(0, total_blocks, [&](int i) {
    const int lblock = i / rrow_blocks;
    const int rblock = i % rrow_blocks;
    const int lrow_begin = lblock * lrows_per_block;
    const int lrow_end = std::min(lrow_begin + lrows_per_block, lrows);
    const int rrow_begin = rblock * rrows_per_block;
    const int rrow_end = std::min(rrow_begin + rrows_per_block, rrows);

    for (int lrow = lrow_begin; lrow < lrow_end; ++lrow) {
      const T* pl = lhs.RowData(lrow);
      T* pout = out.RowData(lrow);
      for (int j = rrow_begin; j < rrow_end; ++j)
        pout[j] = XYT_Kernel(pl, rhs.RowData(j), lcols);
    }
  });
  return out;
}

template <IsVector2D X, IsMatrix Y>
auto MatMul_XYT(const X& lhs, const Y& rhs) {
  using T = decltype(std::declval<typename X::Scalar>() * std::declval<typename Y::Scalar>());
  assert(lhs.Columns() == rhs.Columns());

  const int lcols = lhs.Columns();
  const int rrows = rhs.Rows();
  RowVector<T> out(rrows);

  constexpr int rrows_per_block = 1024;
  const int rrow_blocks = (rrows + rrows_per_block - 1) / rrows_per_block;
  const int total_blocks = rrow_blocks;

  T* dst = out.RowData(0);
  ParallelFor(0, total_blocks, [&](int i) {
  //for (int i = 0; i < total_blocks; ++i) {
    const int rrow_begin = i * rrows_per_block;
    const int rrow_end = std::min(rrow_begin + rrows_per_block, rrows);

    const T* pl = &lhs(0);
    for (int j = rrow_begin; j < rrow_end; ++j)
      dst[j] = XYT_Kernel(pl, rhs.RowData(j), lcols);
  }
  );
  return out;
}

template <IsVector2D X, IsMatrix Y>
auto MatMul_XY(const X& lhs, const Y& rhs) {
  using T = decltype(std::declval<typename X::Scalar>() * std::declval<typename Y::Scalar>());
  assert(lhs.Columns() == rhs.Rows());

  const int lcols = lhs.Columns();
  const int rcols = rhs.Columns();
  RowVector<T> out(rcols);

  constexpr int rcols_per_block = 16 * 32;
  const int rcol_blocks = (rcols + rcols_per_block - 1) / rcols_per_block;
  const int total_blocks = rcol_blocks;
  
  T* dst = out.RowData(0);
  //ParallelFor(0, total_blocks, [&](int block) {
  for (int block = 0; block < total_blocks; ++block) {
    const int rcol_begin = block * rcols_per_block;
    const int rcol_end = std::min(rcol_begin + rcols_per_block, rcols);
    std::fill(dst + rcol_begin, dst + rcol_end, T{0});

    for (int i = 0; i < lcols; ++i) {
      const T li = lhs(i);
      const T* pr = rhs.RowData(i);
      for (int j = rcol_begin; j < rcol_end; ++j)
        dst[j] += li * pr[j];
    }
  }
  //);
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
  requires (!(IsVector<L> && IsVector<R>))
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

template <IsVector L, IsVector R>
auto operator+(const L& lhs, const R& rhs) {
  using T = decltype(std::declval<typename L::Scalar>() + std::declval<typename R::Scalar>());
  assert(lhs.Size() == rhs.Size());
  RowVector<T> out{lhs.Size()};
  for (int i = 0; i < lhs.Size(); ++i)
    out(i) = lhs(i) + rhs(i);
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

template <typename T>
RowVector<T> operator*(RowVector<T> lhs, T rhs) {
  for (int i = 0; i < lhs.Size(); ++i)
    lhs(i) *= rhs;
  return lhs;
}

template <IsVector V, std::floating_point T>
auto operator*(const V& lhs, T rhs) {
  using U = decltype(std::declval<typename V::Scalar>() * T{0});
  RowVector<U> out(lhs.Size());
  for (int i = 0; i < out.Size(); ++i)
    out(i) = lhs(i) * rhs;
  return out;
}

template <IsVector Lhs, IsMatrix Rhs>
auto operator*(const Lhs& lhs, const Rhs& rhs) {
  return MatMul_XY(lhs, rhs);
}