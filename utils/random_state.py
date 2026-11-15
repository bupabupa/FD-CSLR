import pdb
import torch
import random
import numpy as np


class RandomState(object):
    def __init__(self, seed):
        torch.set_num_threads(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def save_rng_state(self):
        rng_dict = {}
        rng_dict["torch"] = torch.get_rng_state()
        rng_dict["cuda"] = torch.cuda.get_rng_state_all()
        numpy_state = np.random.get_state()
        rng_dict["numpy"] = {
            "name": str(numpy_state[0]),
            "keys": torch.from_numpy(
                numpy_state[1].astype(np.int64, copy=True)
            ),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        }
        rng_dict["random"] = random.getstate()
        return rng_dict

    def set_rng_state(self, rng_dict):
        torch.set_rng_state(rng_dict["torch"])
        torch.cuda.set_rng_state_all(rng_dict["cuda"])
        numpy_state = rng_dict["numpy"]
        if isinstance(numpy_state, dict):
            keys = numpy_state["keys"]
            if isinstance(keys, torch.Tensor):
                keys = keys.detach().cpu().numpy().astype(
                    np.uint32, copy=True
                )
            np.random.set_state((
                numpy_state["name"],
                keys,
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            ))
        else:
            # Backward compatibility with checkpoints that stored NumPy's
            # native tuple. New checkpoints use the safe dictionary above.
            np.random.set_state(numpy_state)
        random.setstate(rng_dict["random"])
