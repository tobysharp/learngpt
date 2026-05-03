#include <cassert>
#include <filesystem>
#include <iostream>

#include "matrix.h"
#include "model.h"

int main() {
  constexpr int tokens_to_generate = 40;

  const auto root = std::filesystem::path(__FILE__).parent_path() / "../..";
  const auto model_dir = root / "models/124M";

  const auto hyper_parameters = LoadHyperParameters(model_dir / "hparams.txt");
  const auto prompt = Model<float>::LoadTokenIds(root / "input.txt");
  
  assert(prompt.Size() + tokens_to_generate < hyper_parameters.context_limit);

  const auto model = Model<float>::Load(hyper_parameters, model_dir);

  const auto logits = model.Forward(prompt);

  std::cout << "Done" << std::endl;
}
