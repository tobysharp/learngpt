#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "matrix.h"
#include "pfor.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Scenario {
  std::string_view name;
  int input_columns;
  int output_rows;
};

struct Options {
  std::optional<std::string> case_name;
  double seconds = 0.2;
  std::optional<int> threads;
  std::optional<int> output_block;
};

constexpr Scenario kScenarios[] = {
  {"decode_logits", 768, 50257},
  {"decode_attn_10", 64, 10},
  {"decode_attn_25", 64, 25},
  {"decode_attn_49", 64, 49},
  {"mlp_fc", 768, 3072},
  {"mlp_proj", 3072, 768},
};

uint32_t NextRandom(uint32_t* state) {
  *state = *state * 1664525u + 1013904223u;
  return *state;
}

float RandomFloat(uint32_t* state) {
  constexpr float scale = 1.0f / static_cast<float>(1u << 24);
  return static_cast<float>(NextRandom(state) >> 8) * scale - 0.5f;
}

void Fill(RowVector<float>* row, uint32_t seed) {
  uint32_t state = seed;
  for (int i = 0; i < row->Size(); ++i)
    (*row)(i) = RandomFloat(&state);
}

void Fill(Matrix<float>* matrix, uint32_t seed) {
  uint32_t state = seed;
  for (int i = 0; i < matrix->Rows(); ++i) {
    float* row = matrix->RowData(i);
    for (int j = 0; j < matrix->Columns(); ++j)
      row[j] = RandomFloat(&state);
  }
}

double Checksum(const RowVector<float>& row) {
  return row(0) + row(row.Size() / 2) + row(row.Size() - 1);
}

double FlopsPerIteration(const Scenario& scenario) {
  return static_cast<double>(scenario.output_rows) * (2.0 * scenario.input_columns + 1.0);
}

void MatMulXYTSerial(const RowVector<float>& lhs, const Matrix<float>& rhs, RowVector<float>* out) {
  const int lcols = lhs.Columns();
  for (int j = 0; j < rhs.Rows(); ++j) {
    const float* pr = rhs.RowData(j);
    float sum = 0.0f;
    for (int k = 0; k < lcols; ++k)
      sum += lhs(k) * pr[k];
    (*out)(j) = sum;
  }
}

void MatMulXYTParallel(const RowVector<float>& lhs, const Matrix<float>& rhs, RowVector<float>* out, ThreadPool* pool, int output_block) {
  const int lcols = lhs.Columns();
  const int rrows = rhs.Rows();
  const int blocks = (rrows + output_block - 1) / output_block;
  const float* pl = &lhs(0);
  float* dst = out->RowData(0);
  pool->ParallelFor(0, blocks, [&](int block) {
    const int begin = block * output_block;
    const int end = std::min(begin + output_block, rrows);
    for (int j = begin; j < end; ++j) {
      const float* pr = rhs.RowData(j);
      float sum = 0.0f;
      for (int k = 0; k < lcols; ++k)
        sum += pl[k] * pr[k];
      dst[j] = sum;
    }
  });
}

std::vector<int> BuildThreadCounts(const Options& options) {
  if (options.threads)
    return {*options.threads};
  const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
  std::vector<int> thread_counts;
  for (int threads = 1; threads < static_cast<int>(hw); threads *= 2)
    thread_counts.push_back(threads);
  if (thread_counts.empty() || thread_counts.back() != static_cast<int>(hw))
    thread_counts.push_back(static_cast<int>(hw));
  return thread_counts;
}

std::vector<int> BuildOutputBlocks(const Options& options) {
  if (options.output_block)
    return {*options.output_block};
  return {16, 32, 64, 128, 256, 512, 1024, 2048};
}

std::vector<Scenario> BuildScenarios(const Options& options) {
  std::vector<Scenario> scenarios;
  for (const Scenario& scenario : kScenarios) {
    if (options.case_name && *options.case_name != scenario.name)
      continue;
    scenarios.push_back(scenario);
  }
  return scenarios;
}

void PrintUsage(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [--case NAME] [--seconds S] [--threads N] [--output-block N]\n";
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
    } else if (arg == "--seconds") {
      options.seconds = std::atof(require_value(arg));
    } else if (arg == "--threads") {
      options.threads = std::atoi(require_value(arg));
    } else if (arg == "--output-block") {
      options.output_block = std::atoi(require_value(arg));
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + std::string(arg));
    }
  }
  return options;
}

void RunScenario(const Scenario& scenario, const Options& options) {
  RowVector<float> lhs{scenario.input_columns};
  Matrix<float> rhs{scenario.output_rows, scenario.input_columns};
  RowVector<float> out{scenario.output_rows};
  Fill(&lhs, 0x12345678u + static_cast<uint32_t>(scenario.input_columns));
  Fill(&rhs, 0x87654321u + static_cast<uint32_t>(scenario.output_rows));

  std::cout << "Scenario " << scenario.name << " in=" << scenario.input_columns
            << " out=" << scenario.output_rows << '\n';

  {
    volatile double sink = 0.0;
    MatMulXYTSerial(lhs, rhs, &out);
    sink += Checksum(out);

    int iterations = 0;
    const auto start = Clock::now();
    auto now = start;
    do {
      MatMulXYTSerial(lhs, rhs, &out);
      sink += Checksum(out);
      ++iterations;
      now = Clock::now();
    } while (std::chrono::duration<double>(now - start).count() < options.seconds);

    const double elapsed = std::chrono::duration<double>(now - start).count();
    const double per_iteration_ms = elapsed * 1000.0 / iterations;
    const double gflops = FlopsPerIteration(scenario) * iterations / elapsed / 1e9;
    std::cout << "  serial"
              << " iter=" << std::setw(6) << iterations
              << " ms/iter=" << std::setw(9) << std::fixed << std::setprecision(3) << per_iteration_ms
              << " gflops=" << std::setw(8) << std::setprecision(2) << gflops
              << " checksum=" << std::setprecision(6) << sink << '\n';
  }

  const std::vector<int> thread_counts = BuildThreadCounts(options);
  const std::vector<int> output_blocks = BuildOutputBlocks(options);
  for (int threads : thread_counts) {
    for (int output_block : output_blocks) {
      ThreadPool pool{threads};
      volatile double sink = 0.0;
      MatMulXYTParallel(lhs, rhs, &out, &pool, output_block);
      sink += Checksum(out);

      int iterations = 0;
      const auto start = Clock::now();
      auto now = start;
      do {
        MatMulXYTParallel(lhs, rhs, &out, &pool, output_block);
        sink += Checksum(out);
        ++iterations;
        now = Clock::now();
      } while (std::chrono::duration<double>(now - start).count() < options.seconds);

      const double elapsed = std::chrono::duration<double>(now - start).count();
      const double per_iteration_ms = elapsed * 1000.0 / iterations;
      const double gflops = FlopsPerIteration(scenario) * iterations / elapsed / 1e9;
      std::cout << "  threads=" << std::setw(2) << threads
                << " output_block=" << std::setw(4) << output_block
                << " iter=" << std::setw(6) << iterations
                << " ms/iter=" << std::setw(9) << std::fixed << std::setprecision(3) << per_iteration_ms
                << " gflops=" << std::setw(8) << std::setprecision(2) << gflops
                << " checksum=" << std::setprecision(6) << sink << '\n';
    }
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
      RunScenario(scenario, options);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    PrintUsage(argv[0]);
    return 1;
  }
}