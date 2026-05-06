#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "matrix.h"
#include "model.h"

int main() {
  constexpr int tokens_to_generate = 40;

  const auto root = std::filesystem::path(__FILE__).parent_path() / "../..";
  const auto model_dir = root / "models/124M";

  const auto hyper_parameters = LoadHyperParameters(model_dir / "hparams.txt");
  auto prompt = LoadTokenIds(root / "input.txt");

  assert(std::ssize(prompt) + tokens_to_generate < hyper_parameters.context_limit);

  const auto model = Model<float>::Load(hyper_parameters, model_dir);

  std::vector<TokenId> generated_tokens;
  generated_tokens.reserve(tokens_to_generate);

  auto start = std::chrono::steady_clock::now();
  auto [logits, cache] = model.Prefill(prompt);
  TokenId token = ArgMax(Row(logits, -1));
  const auto prefill_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < tokens_to_generate; ++i) {
    generated_tokens.push_back(token);
    token = ArgMax(model.Decode(token, &cache));
  }
  const auto decode_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();

  std::ofstream f(root / "output.txt");
  for (TokenId generated_token : generated_tokens)
    f << generated_token << std::endl;

  std::cout << "Prefill: " << prefill_ms << " ms, per token: " << prefill_ms / std::ssize(prompt) << " ms." << std::endl;
  std::cout << "Decode: " << decode_ms << " ms, per token: " << decode_ms / tokens_to_generate << " ms." << std::endl;
  std::cout << "Total: " << prefill_ms + decode_ms << " ms." << std::endl;
}
