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

#include "layers.h"

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
  double seconds = 0.25;
  bool list_only = false;
};

constexpr Shape kShapes[] = {
  {"attn_qkv", 768, 2304},
  {"attn_proj", 768, 768},
  {"mlp_fc", 768, 3072},
  {"mlp_proj", 3072, 768},
};

constexpr int kDefaultRows[] = {10, 25, 49};

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
    float* row = (*matrix)[i];
    for (int j = 0; j < matrix->Columns(); ++j)
      row[j] = RandomFloat(&state);
  }
}

void Fill(RowVector<float>* vector, uint32_t seed) {
  uint32_t state = seed;
  for (int i = 0; i < vector->Size(); ++i)
    (*vector)[i] = RandomFloat(&state);
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
    << "Usage: " << argv0 << " [--case NAME] [--rows N] [--seconds S] [--list]\n"
    << "\n"
    << "Relevant cases:\n";
  for (const Shape& shape : kShapes)
    std::cout << "  " << shape.name << "  (N x " << shape.input_columns << ") * (" << shape.input_columns << " x " << shape.output_columns << ")\n";
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
    } else if (arg == "--list") {
      options.list_only = true;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + std::string(arg));
    }
  }
  if (options.rows && *options.rows <= 0)
    throw std::runtime_error("--rows must be positive");
  if (!(options.seconds > 0.0))
    throw std::runtime_error("--seconds must be positive");
  return options;
}

double FlopsPerIteration(const Scenario& scenario) {
  return static_cast<double>(scenario.rows) * scenario.output_columns * (2.0 * scenario.input_columns + 1.0);
}

double Checksum(const Matrix<float>& matrix) {
  double sum = 0.0;
  for (int i = 0; i < matrix.Rows(); ++i) {
    const float* row = matrix[i];
    sum += row[0];
    sum += row[matrix.Columns() - 1];
  }
  return sum;
}

void RunScenario(const Scenario& scenario, double seconds) {
  Affine<float> affine{scenario.input_columns, scenario.output_columns};
  Matrix<float> input{scenario.rows, scenario.input_columns};
  Fill(&affine.weights_T, 0x12345678u + static_cast<uint32_t>(scenario.rows + scenario.output_columns));
  Fill(&affine.bias, 0x87654321u + static_cast<uint32_t>(scenario.input_columns));
  Fill(&input, 0x31415926u + static_cast<uint32_t>(scenario.rows));

  volatile double sink = 0.0;
  sink += Checksum(affine(input));

  int iterations = 0;
  const auto start = Clock::now();
  auto now = start;
  do {
    const Matrix<float> output = affine(input);
    sink += Checksum(output);
    ++iterations;
    now = Clock::now();
  } while (std::chrono::duration<double>(now - start).count() < seconds);

  const double elapsed = std::chrono::duration<double>(now - start).count();
  const double per_iteration_ms = elapsed * 1000.0 / iterations;
  const double gflops = FlopsPerIteration(scenario) * iterations / elapsed / 1e9;

  std::cout << std::left << std::setw(12) << scenario.name
            << " rows=" << std::setw(3) << scenario.rows
            << " in=" << std::setw(4) << scenario.input_columns
            << " out=" << std::setw(5) << scenario.output_columns
            << " iter=" << std::setw(6) << iterations
            << " total_ms=" << std::setw(9) << std::fixed << std::setprecision(3) << elapsed * 1000.0
            << " ms/iter=" << std::setw(9) << per_iteration_ms
            << " gflops=" << std::setw(8) << std::setprecision(2) << gflops
            << " checksum=" << std::setprecision(6) << sink
            << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseOptions(argc, argv);
    const std::vector<Scenario> scenarios = BuildScenarios(options);
    if (options.list_only) {
      for (const Scenario& scenario : scenarios)
        std::cout << scenario.name << " rows=" << scenario.rows << " in=" << scenario.input_columns << " out=" << scenario.output_columns << '\n';
      return 0;
    }
    if (scenarios.empty()) {
      std::cerr << "No scenarios matched the requested filters.\n";
      return 1;
    }

    std::cout << "Benchmarking Affine<float>::operator() with row-major activations and row-major weights.\n";
    for (const Scenario& scenario : scenarios)
      RunScenario(scenario, options.seconds);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    PrintUsage(argv[0]);
    return 1;
  }
}