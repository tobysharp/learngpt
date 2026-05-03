#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "matrix.h"
#include "model.h"

int main() {
  constexpr int tokens_to_generate = 40;

  const auto root = std::filesystem::path(__FILE__).parent_path() / "../..";
  const auto model_dir = root / "models/124M";

  const auto hyper_parameters = LoadHyperParameters(model_dir / "hparams.txt");
  auto tokens = LoadTokenIds(root / "input.txt");
  const int prompt_size = std::ssize(tokens);

  assert(std::ssize(tokens) + tokens_to_generate < hyper_parameters.context_limit);

  const auto model = Model<float>::Load(hyper_parameters, model_dir);

  std::ofstream f(root / "output.txt");
  for (int i = 0; i < tokens_to_generate; ++i) {
    const auto logits = model.Forward(tokens);
    const TokenId next_id = ArgMax(Row(logits, logits.Rows() - 1));
    f << next_id << std::endl;
    tokens.push_back(next_id);
  }
}
