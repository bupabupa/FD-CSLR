import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from scripts.build_text_embeddings import (
    lexical_descriptions,
    write_embeddings,
)


def test_lexical_description_backend_uses_class_ids():
    gloss_dict = {"a": [1, 0], "b": [2, 1]}
    embeddings = lexical_descriptions(
        gloss_dict,
        num_classes=3,
        output_dim=16,
        descriptions={"a": "alpha definition", "b": "beta definition"},
    )
    assert embeddings.shape == (3, 16)
    as_tensor = (
        embeddings
        if torch.is_tensor(embeddings)
        else torch.from_numpy(embeddings)
    )
    assert torch.count_nonzero(as_tensor[0]) == 0
    assert not torch.allclose(
        as_tensor[1],
        as_tensor[2],
    )


def test_embedding_metadata_records_content_hash():
    embeddings = np.zeros((3, 8), dtype=np.float32)
    embeddings[1, 0] = 1.0
    embeddings[2, 1] = 1.0
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "text.npy"
        metadata = Path(directory) / "text.metadata.json"
        write_embeddings(
            output,
            embeddings,
            metadata,
            {"backend": "unit-test"},
        )
        payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["shape"] == [3, 8]
    assert len(payload["output_sha256"]) == 64
    assert payload["backend"] == "unit-test"


if __name__ == "__main__":
    test_lexical_description_backend_uses_class_ids()
    test_embedding_metadata_records_content_hash()
    print("text embedding builder tests passed")
