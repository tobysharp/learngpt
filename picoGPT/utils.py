import json
import os
import re

import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm

from encoder import get_encoder


def iter_gpt2_parameter_tensors(params):
    yield "wte", params["wte"]
    yield "wpe", params["wpe"]

    for index, block in enumerate(params["blocks"]):
        prefix = f"blocks.{index}"
        yield f"{prefix}.ln_1.g", block["ln_1"]["g"]
        yield f"{prefix}.ln_1.b", block["ln_1"]["b"]
        yield f"{prefix}.attn.c_attn.w", block["attn"]["c_attn"]["w"]
        yield f"{prefix}.attn.c_attn.b", block["attn"]["c_attn"]["b"]
        yield f"{prefix}.attn.c_proj.w", block["attn"]["c_proj"]["w"]
        yield f"{prefix}.attn.c_proj.b", block["attn"]["c_proj"]["b"]
        yield f"{prefix}.ln_2.g", block["ln_2"]["g"]
        yield f"{prefix}.ln_2.b", block["ln_2"]["b"]
        yield f"{prefix}.mlp.c_fc.w", block["mlp"]["c_fc"]["w"]
        yield f"{prefix}.mlp.c_fc.b", block["mlp"]["c_fc"]["b"]
        yield f"{prefix}.mlp.c_proj.w", block["mlp"]["c_proj"]["w"]
        yield f"{prefix}.mlp.c_proj.b", block["mlp"]["c_proj"]["b"]

    yield "ln_f.g", params["ln_f"]["g"]
    yield "ln_f.b", params["ln_f"]["b"]


def _tensor_file_name(name):
    return name.replace(".", "_") + ".bin"


def export_gpt2_params(params, output_dir, write_metadata=True):
    os.makedirs(output_dir, exist_ok=True)

    metadata_lines = [
        "# Raw float32 tensor files written in NumPy C-order (row-major), little-endian.",
        "# Format: filename dim0 dim1 ...",
    ]

    for name, tensor in iter_gpt2_parameter_tensors(params):
        array = np.asarray(tensor, dtype="<f4")
        file_name = _tensor_file_name(name)
        array.tofile(os.path.join(output_dir, file_name))
        shape_suffix = " ".join(str(dimension) for dimension in array.shape)
        metadata_lines.append(f"{file_name} {shape_suffix}".rstrip())

    if write_metadata:
        with open(os.path.join(output_dir, "metadata.txt"), "w") as f:
            f.write("\n".join(metadata_lines) + "\n")


def download_gpt2_files(model_size, model_dir):
    assert model_size in ["124M", "355M", "774M", "1558M"]
    for filename in [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]:
        url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
        r = requests.get(f"{url}/{model_size}/{filename}", stream=True)
        r.raise_for_status()

        with open(os.path.join(model_dir, filename), "wb") as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(
                ncols=100,
                desc="Fetching " + filename,
                total=file_size,
                unit_scale=True,
                unit="b",
            ) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)


def load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams):
    def set_in_nested_dict(d, keys, val):
        if not keys:
            return val
        if keys[0] not in d:
            d[keys[0]] = {}
        d[keys[0]] = set_in_nested_dict(d[keys[0]], keys[1:], val)
        return d

    params = {"blocks": [{} for _ in range(hparams["n_layer"])]}
    for name, _ in tf.train.list_variables(tf_ckpt_path):
        array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
        name = name[len("model/") :]
        if name.startswith("h"):
            m = re.match(r"h([0-9]+)/(.*)", name)
            n = int(m[1])
            sub_name = m[2]
            set_in_nested_dict(params["blocks"][n], sub_name.split("/"), array)
        else:
            set_in_nested_dict(params, name.split("/"), array)

    return params


def load_encoder_hparams_and_params(model_size, models_dir):
    assert model_size in ["124M", "355M", "774M", "1558M"]

    model_dir = os.path.join(models_dir, model_size)
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    if not tf_ckpt_path:  # download files if necessary
        os.makedirs(model_dir, exist_ok=True)
        download_gpt2_files(model_size, model_dir)
        tf_ckpt_path = tf.train.latest_checkpoint(model_dir)

    encoder = get_encoder(model_size, models_dir)
    hparams = json.load(open(os.path.join(model_dir, "hparams.json")))
    with open(os.path.join(model_dir, "hparams.txt"), "w") as f:
        for name in hparams:
            f.write(f"{name}: {hparams[name]}\n")
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams)
    export_gpt2_params(params, model_dir)

    return encoder, hparams, params
