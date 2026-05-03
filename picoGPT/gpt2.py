from pathlib import Path

import numpy as np


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + eps)  # normalize x to have mean=0 and var=1 over last axis
    return g * x + b  # scale and offset with gamma/beta params


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b


def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # project up
    a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]

    # project back down
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

    return x


def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projection
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    qkv = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

    # split into heads
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]

    # causal mask to hide future inputs from being attended to
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]

    # perform attention over each head
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]  # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]

    # merge heads
    x = np.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # multi-head causal self attention
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):  # [n_seq] -> [n_seq, n_vocab]
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]

    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, **params, n_head=n_head)  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids


def decode_output_file(
    token_ids_path: str | None = None,
    input_ids_path: str | None = None,
    model_size: str = "124M",
    models_dir: str | None = None,
):
    from utils import load_encoder_hparams_and_params

    repo_root = Path(__file__).resolve().parent.parent
    resolved_token_ids_path = Path(token_ids_path) if token_ids_path else repo_root / "output.txt"
    resolved_input_ids_path = Path(input_ids_path) if input_ids_path else repo_root / "input.txt"
    resolved_models_dir = Path(models_dir) if models_dir else repo_root / "models"

    with resolved_input_ids_path.open("r", encoding="ascii") as file_handle:
        input_ids = [int(line.strip()) for line in file_handle if line.strip()]

    with resolved_token_ids_path.open("r", encoding="ascii") as file_handle:
        expected_output_ids = [int(line.strip()) for line in file_handle if line.strip()]

    encoder, hparams, params = load_encoder_hparams_and_params(model_size, str(resolved_models_dir))
    assert len(input_ids) + len(expected_output_ids) < hparams["n_ctx"]

    generated_output_ids = generate(input_ids.copy(), params, hparams["n_head"], len(expected_output_ids))

    if generated_output_ids != expected_output_ids:
        mismatch_index = next(
            index
            for index, (generated_id, expected_id) in enumerate(zip(generated_output_ids, expected_output_ids))
            if generated_id != expected_id
        )
        raise ValueError(
            "Generated tokens do not match output.txt at index "
            f"{mismatch_index}: expected {expected_output_ids[mismatch_index]}, got {generated_output_ids[mismatch_index]}"
        )

    print(f"Validated {len(expected_output_ids)} generated tokens: match")

    return encoder.decode(generated_output_ids)


def main(
    prompt: str,
    n_tokens_to_generate: int = 40,
    model_size: str = "124M",
    models_dir: str = "models",
):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)
    print(input_ids)
    with open("input.txt", "w") as f:
        for id in input_ids:
            f.write(f"{id}\n")

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    print(output_ids)

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)

    return output_text


if __name__ == "__main__":
    import fire

    class GPT2CLI:
        def __call__(
            self,
            prompt: str,
            n_tokens_to_generate: int = 40,
            model_size: str = "124M",
            models_dir: str = "models",
        ):
            return main(prompt, n_tokens_to_generate, model_size, models_dir)

        def decode_output_file(
            self,
            token_ids_path: str | None = None,
            input_ids_path: str | None = None,
            model_size: str = "124M",
            models_dir: str | None = None,
        ):
            return decode_output_file(token_ids_path, input_ids_path, model_size, models_dir)

    fire.Fire(GPT2CLI)
