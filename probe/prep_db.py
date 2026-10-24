"""Build NCRL's retrieval DB (dataset_db.npz) for selected benchmarks.

Same computation as ncrl_src/src/prepare_dataset_kv.py (same encoder config, first frame of every trajectory,
features in chunks of 100, same keys 'dmc'/'mw' with 'filename' and 'feature'), but the benchmarks to process are
an argument, so the Meta-World DB can be built before the DMC download finishes. Finetune's _retrieve_buffer only
reads the entry of the task's own benchmark (env_name prefix), so a Meta-World-only DB serves Meta-World tasks.

  python -I prep_db.py --ncrl /home/yeho/RL/ncrl_src --model-basedir /home/yeho/RL/ncrl_data/models/models \
      --data-basedir /home/yeho/RL/ncrl_data/ncrl_datasets --benchmarks metaworld --out /home/yeho/RL/ncrl_data/dataset_db_mw.npz
"""
import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ncrl", required=True)
    p.add_argument("--model-basedir", type=Path, required=True)
    p.add_argument("--data-basedir", type=Path, required=True)
    p.add_argument("--benchmarks", default="metaworld", help="comma list of dmcontrol,metaworld")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, a.ncrl + "/src")
    from omegaconf import OmegaConf
    from tqdm import tqdm
    import dreamer.nets as nets

    device = torch.device("cuda")
    cfg = OmegaConf.create({"cnn_depth": 96, "use_norm": True, "cnn_kernels": [4, 4, 4, 4], "act": "SiLU"})
    encoder = nets.Encoder((3, 64, 64), **cfg)
    db, t0 = {}, time.time()
    for benchmark in a.benchmarks.split(","):
        filename = list((a.data_basedir / benchmark).rglob("*.npz"))
        print(f"{benchmark}: {len(filename)} trajectories", flush=True)
        param_dict = torch.load(a.model_basedir / f"{benchmark}.pt", weights_only=False)
        encoder.load_state_dict(param_dict["wm"]["encoder"])
        encoder.to(device).eval()
        # .copy(): NCRL's script keeps data['observation'][0], a view that holds the whole episode array in memory
        # (10k DMC episodes ~ 60 GB). Copying the first frame gives the same values without that.
        first_frames = [np.load(fn)["observation"][0].copy() for fn in tqdm(filename)]
        first_frames = (torch.from_numpy(np.stack(first_frames)).float() / 255. - 0.5).to(device)
        with torch.no_grad():
            feats = np.concatenate([encoder(first_frames[i:i + 100]).cpu().numpy()
                                    for i in range(0, first_frames.size(0), 100)], axis=0)
        db[{"dmcontrol": "dmc", "metaworld": "mw"}[benchmark]] = {"filename": np.array(filename), "feature": feats}
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **db)
        bs.seek(0)
        a.out.write_bytes(bs.read())
    print(f"saved {a.out} in {time.time() - t0:.0f}s: " + ", ".join(f"{k} {v['feature'].shape}" for k, v in db.items()))


if __name__ == "__main__":
    main()
