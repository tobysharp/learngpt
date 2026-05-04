#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "matrix.h"
#include "pfor.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Shape {
  std::string_view name;
  int input_columns;
  int output_columns;
};

struct Scenario {
  std::string name;
  int rows;
  int input_columns;
  int output_columns;
};

struct Options {
  std::optional<std::string> case_name;
  std::optional<int> rows;
  double seconds = 0.2;
  int rrows_per_block = 64;
};

constexpr Shape kShapes[] = {
  {"attn_qkv", 768, 2304},
  {"attn_proj", 768, 768},
  {"mlp_fc", 768, 3072},
  {"mlp_proj", 3072, 768},
};

constexpr int kDefaultRows[] = {10, 49};

uint32_t NextRandom(uint32_t* state) {
  *state = *state * 1664525u + 1013904223u;
  return *state;
}

float RandomFloat(uint32_t* state) {
  constexpr float scale = 1.0f / static_cast<float>(1u << 24);
  return static_cast<float>(NextRandom(state) >> 8) * scale - 0.5f;
}

void Fill(Matrix<float>* matrix, uint32_t seed) {
  uint32_t state = seed;
  for (int i = 0; i < matrix->Rows(); ++i) {
    float* row = matrix->RowData(i);
    for (int j = 0; j < matrix->Columns(); ++j)
      row[j] = RandomFloat(&state);
  }
}

double Checksum(const Matrix<float>& matrix) {
  double sum = 0.0;
  for (int i = 0; i < matrix.Rows(); ++i) {
    const float* row = matrix.RowData(i);
    sum += row[0];
    sum += row[matrix.Columns() - 1];
  }
  return sum;
}

double FlopsPerIteration(const Scenario& scenario) {
  return static_cast<double>(scenario.rows) * scenario.output_columns * (2.0 * scenario.input_columns + 1.0);
}

void MatMulXYTParallel(const Matrix<float>& lhs, const Matrix<float>& rhs, Matrix<float>* out, ThreadPool* pool, int rrows_per_block) {
  const int lrows = lhs.Rows();
  const int lcols = lhs.Columns();
  const int rrows = rhs.Rows();
  const int blocks_per_lrow = (rrows + rrows_per_block - 1) / rrows_per_block;
  const int total_blocks = lrows * blocks_per_lrow;

  pool->ParallelFor(0, total_blocks, [&](int index) {
    const int lrow = index / blocks_per_lrow;
    const int block_index = index % blocks_per_lrow;
    const int rrow_begin = block_index * rrows_per_block;
    const int rrow_end = std::min(rrow_begin + rrows_per_block, rrows);
    const float* pl = lhs.RowData(lrow);
    float* pout = out->RowData(lrow);
    for (int j = rrow_begin; j < rrow_end; ++j) {
      const float* pr = rhs.RowData(j);
      float sum = 0.0f;
      for (int k = 0; k < lcols; ++k)
        sum += pl[k] * pr[k];
      pout[j] = sum;
    }
  });
}

std::vector<Scenario> BuildScenarios(const Options& options) {
  std::vector<Scenario> scenarios;
  for (const Shape& shape : kShapes) {
    if (options.case_name && *options.case_name != shape.name)
      continue;
    if (options.rows) {
      scenarios.push_back(Scenario{std::string(shape.name), *options.rows, shape.input_columns, shape.output_columns});
      continue;
    }
    for (int rows : kDefaultRows)
      scenarios.push_back(Scenario{std::string(shape.name), rows, shape.input_columns, shape.output_columns});
  }
  return scenarios;
}

void PrintUsage(const char* argv0) {
  std::cout
    << "Usage: " << argv0 << " [--case NAME] [--rows N] [--seconds S] [--rrows-per-block N]\n";
}

Options ParseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    auto require_value = [&](std::string_view flag) -> const char* {
      if (i + 1 >= argc)
        throw std::runtime_error("missing value for " + std::string(flag));
      return argv[++i];
    };
    if (arg == "--case") {
      options.case_name = require_value(arg);
    } else if (arg == "--rows") {
      options.rows = std::atoi(require_value(arg));
    } else if (arg == "--seconds") {
      options.seconds = std::atof(require_value(arg));
    } else if (arg == "--rrows-per-block") {
      options.rrows_per_block = std::atoi(require_value(arg));
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + std::string(arg));
    }
  }
  return options;
}

void RunScenario(const Scenario& scenario, double seconds, int rrows_per_block) {
  Matrix<float> lhs{scenario.rows, scenario.input_columns};
  Matrix<float> rhs{scenario.output_columns, scenario.input_columns};
  Matrix<float> out{scenario.rows, scenario.output_columns};
  Fill(&lhs, 0x12345678u + static_cast<uint32_t>(scenario.rows));
  Fill(&rhs, 0x87654321u + static_cast<uint32_t>(scenario.output_columns));

  const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
  std::vector<int> thread_counts;
  for (int threads = 1; threads < static_cast<int>(hw); threads *= 2)
    thread_counts.push_back(threads);
  if (thread_counts.empty() || thread_counts.back() != static_cast<int>(hw))
    thread_counts.push_back(static_cast<int>(hw));

  std::cout << "Scenario " << scenario.name << " rows=" << scenario.rows << " in=" << scenario.input_columns
            << " out=" << scenario.output_columns << " rrows_per_block=" << rrows_per_block << '\n';
  for (int threads : thread_counts) {
    ThreadPool pool{threads};
    volatile double sink = 0.0;
    MatMulXYTParallel(lhs, rhs, &out, &pool, rrows_per_block);
    sink += Checksum(out);

    int iterations = 0;
    const auto start = Clock::now();
    auto now = start;
    do {
      MatMulXYTParallel(lhs, rhs, &out, &pool, rrows_per_block);
      sink += Checksum(out);
      ++iterations;
      now = Clock::now();
    } while (std::chrono::duration<double>(now - start).count() < seconds);

    const double elapsed = std::chrono::duration<double>(now - start).count();
    const double per_iteration_ms = elapsed * 1000.0 / iterations;
    const double gflops = FlopsPerIteration(scenario) * iterations / elapsed / 1e9;
    std::cout << "  threads=" << std::setw(2) << threads
              << " iter=" << std::setw(6) << iterations
              << " ms/iter=" << std::setw(9) << std::fixed << std::setprecision(3) << per_iteration_ms
              << " gflops=" << std::setw(8) << std::setprecision(2) << gflops
              << " checksum=" << std::setprecision(6) << sink << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    const std::vector<Scenario> scenarios = BuildScenarios(options);
    if (scenarios.empty()) {
      std::cerr << "No scenarios matched the requested filters.\n";
      return 1;
    }
    for (const Scenario& scenario : scenarios)
      RunScenario(scenario, options.seconds, options.rrows_per_block);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    PrintUsage(argv[0]);
    return 1;
  }
}