"""Check that a dataset can be trained: load it through RecBole, run a few real training steps, extrapolate the epoch.

e.g.  python scripts/check_data.py -d Amazon_Beauty --steps 30
"""

import argparse
import json
import resource
import time

import torch

from freqmat.compat import allow_unsafe_torch_load
from freqmat.config import build_config, parse_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-m", "--model", default="FreqMaT")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("-s", "--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    allow_unsafe_torch_load()
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import init_seed

    from freqmat.models import resolve_model

    config = build_config(args.model, args.dataset, parse_overrides(args.set))
    init_seed(config["seed"], config["reproducibility"])
    t0 = time.time()
    data = create_dataset(config)
    train, valid, test = data_preparation(config, data)
    load_s = time.time() - t0
    net = resolve_model(args.model)(config, train.dataset).to(config["device"])
    params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=config["learning_rate"])
    net.train()
    times = []
    for i, batch in enumerate(train):
        if i >= args.steps:
            break
        torch.cuda.synchronize()
        t = time.time()
        loss = net.calculate_loss(batch.to(config["device"]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - t)
        assert torch.isfinite(loss), f"non-finite loss at step {i}"
    train_peak = torch.cuda.max_memory_allocated() / 2**30
    train_reserved = torch.cuda.max_memory_reserved() / 2**30
    net.eval()
    with torch.no_grad():  # the same code path as validation / test evaluation (full ranking over all items)
        for loader in (valid, test):
            for i, batch in enumerate(loader):
                if i >= args.steps:
                    break
                interaction = batch[0].to(config["device"])
                scores = net.full_sort_predict(interaction)
                assert scores.shape[-1] == data.item_num, "full-sort scores must cover every item"
    eval_peak = torch.cuda.max_memory_allocated() / 2**30
    step_s = sorted(times[5:] or times)[len(times[5:] or times) // 2]
    steps_per_epoch = len(train)
    report = {
        "dataset": args.dataset,
        "users": data.user_num - 1,
        "items": data.item_num - 1,
        "interactions": int(data.inter_num),
        "train_batches": steps_per_epoch,
        "batch_size": config["train_batch_size"],
        "max_len": config["MAX_ITEM_LIST_LENGTH"],
        "params_M": round(params / 1e6, 2),
        "load_s": round(load_s, 1),
        "step_s": round(step_s, 3),
        "epoch_min": round(step_s * steps_per_epoch / 60, 1),
        "train_peak_gpu_gb": round(train_peak, 2),
        "eval_peak_gpu_gb": round(eval_peak, 2),
        "reserved_gpu_gb": round(max(train_reserved, torch.cuda.max_memory_reserved() / 2**30), 2),
        "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "peak_ram_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
        "train_samples": len(train.dataset),
        "valid_samples": len(valid.dataset),
        "test_samples": len(test.dataset),
        "final_loss": round(float(loss), 3),
    }
    print("CHECK " + json.dumps(report))


if __name__ == "__main__":
    main()
