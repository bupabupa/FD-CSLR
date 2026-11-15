import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.text_prototype import (  # noqa: E402
    _lexical_features,
    build_lexical_embeddings,
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _class_texts(gloss_dict):
    return {
        str(gloss): " ".join(str(gloss).replace("_", " ").split())
        for gloss in gloss_dict
    }


def wordnet_descriptions(gloss_dict, language="eng", max_synsets=3):
    try:
        from nltk.corpus import wordnet as wn
    except ImportError as error:
        raise RuntimeError(
            "WordNet backend requires nltk and the WordNet corpus"
        ) from error
    output = {}
    for gloss in gloss_dict:
        query = str(gloss).replace("_", " ")
        synsets = wn.synsets(query, lang=language)[: int(max_synsets)]
        descriptions = []
        for synset in synsets:
            description = synset.definition()
            if description:
                descriptions.append(description)
        output[str(gloss)] = " ".join(
            [query] + descriptions
        ).strip()
    return output


def load_descriptions(path):
    if path is None:
        return None
    with open(path, encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError("description JSON must map gloss strings to text")
    return {str(key): str(value) for key, value in payload.items()}


def lexical_descriptions(gloss_dict, num_classes, output_dim, descriptions=None):
    embeddings = build_lexical_embeddings(
        gloss_dict,
        num_classes=num_classes,
        output_dim=output_dim,
    )
    if descriptions is None:
        return embeddings
    for gloss, value in gloss_dict.items():
        text = descriptions.get(str(gloss))
        if text:
            class_id = int(value[0])
            embeddings[class_id] = _lexical_features(
                text,
                output_dim,
                ngram_min=1,
                ngram_max=5,
            )
    return embeddings


def llama_descriptions(
    gloss_dict,
    descriptions,
    model_name,
    device,
    layer=-1,
    pooling="mean",
    batch_size=8,
    dtype="float16",
):
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Llama backend requires transformers and a local/cached model"
        ) from error
    if pooling not in {"mean", "last"}:
        raise ValueError("pooling must be mean or last")
    if dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("unsupported torch dtype")
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()
    glosses = list(gloss_dict.keys())
    texts = [
        descriptions.get(str(gloss), _class_texts(gloss_dict)[str(gloss)])
        for gloss in glosses
    ]
    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), int(batch_size)):
            batch = texts[start:start + int(batch_size)]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = output.hidden_states[int(layer)]
            if pooling == "mean":
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            else:
                lengths = encoded["attention_mask"].sum(dim=1) - 1
                pooled = hidden[
                    torch.arange(hidden.size(0), device=hidden.device),
                    lengths,
                ]
            vectors.append(F.normalize(pooled.float(), dim=-1).cpu())
    matrix = torch.cat(vectors, dim=0).numpy()
    embeddings = np.zeros(
        (len(gloss_dict) + 1, matrix.shape[1]),
        dtype=np.float32,
    )
    for index, gloss in enumerate(glosses):
        embeddings[int(gloss_dict[gloss][0])] = matrix[index]
    return embeddings


def write_embeddings(output, embeddings, metadata_path, metadata):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, np.asarray(embeddings, dtype=np.float32))
    metadata = dict(metadata)
    metadata["output"] = str(output.resolve())
    metadata["output_sha256"] = _sha256(output)
    metadata["shape"] = list(np.asarray(embeddings).shape)
    metadata["created_utc"] = datetime.now(timezone.utc).isoformat()
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output, metadata_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gloss-dict", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument(
        "--backend",
        choices=["lexical", "wordnet+lexical", "llama", "wordnet+llama"],
        default="lexical",
    )
    parser.add_argument("--descriptions-json", default=None)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--wordnet-language", default="eng")
    parser.add_argument("--wordnet-max-synsets", type=int, default=3)
    parser.add_argument("--llama-model", default=None)
    parser.add_argument("--llama-layer", type=int, default=-1)
    parser.add_argument("--llama-pooling", choices=["mean", "last"], default="mean")
    parser.add_argument("--llama-batch-size", type=int, default=8)
    parser.add_argument(
        "--llama-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    gloss_dict = np.load(
        args.gloss_dict,
        allow_pickle=True,
    ).item()
    num_classes = len(gloss_dict) + 1
    descriptions = load_descriptions(args.descriptions_json)
    if args.backend.startswith("wordnet"):
        wordnet = wordnet_descriptions(
            gloss_dict,
            language=args.wordnet_language,
            max_synsets=args.wordnet_max_synsets,
        )
        if descriptions is None:
            descriptions = wordnet
        else:
            descriptions = {
                gloss: " ".join(
                    filter(
                        None,
                        [descriptions.get(gloss, ""), wordnet.get(gloss, "")],
                    )
                )
                for gloss in gloss_dict
            }

    if args.backend.endswith("lexical"):
        embeddings = lexical_descriptions(
            gloss_dict,
            num_classes,
            args.embedding_dim,
            descriptions=descriptions,
        )
    else:
        if not args.llama_model:
            raise ValueError("Llama backend requires --llama-model")
        embeddings = llama_descriptions(
            gloss_dict,
            descriptions or _class_texts(gloss_dict),
            model_name=args.llama_model,
            device=args.device,
            layer=args.llama_layer,
            pooling=args.llama_pooling,
            batch_size=args.llama_batch_size,
            dtype=args.llama_dtype,
        )

    metadata_path = args.metadata_output or str(
        Path(args.output).with_suffix(".metadata.json")
    )
    metadata = {
        "backend": args.backend,
        "gloss_dict": str(Path(args.gloss_dict).resolve()),
        "gloss_dict_sha256": _sha256(args.gloss_dict),
        "num_classes": num_classes,
        "wordnet_language": (
            args.wordnet_language
            if args.backend.startswith("wordnet")
            else None
        ),
        "wordnet_max_synsets": (
            args.wordnet_max_synsets
            if args.backend.startswith("wordnet")
            else None
        ),
        "llama_model": args.llama_model,
        "llama_layer": args.llama_layer if "llama" in args.backend else None,
        "llama_pooling": (
            args.llama_pooling if "llama" in args.backend else None
        ),
        "normalization": (
            "L2" if args.backend != "lexical" or descriptions else "L2"
        ),
    }
    output, metadata_path = write_embeddings(
        args.output,
        embeddings,
        metadata_path,
        metadata,
    )
    print("embeddings: {}".format(output))
    print("metadata: {}".format(metadata_path))


if __name__ == "__main__":
    main()
