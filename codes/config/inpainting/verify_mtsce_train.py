"""Short training smoke test for the MTSCE integration (stage-1 warm-up path).

It exercises the real training stack -- option parsing, LQGTMask dataset,
DenoisingModel (weight loading, freeze, parameter groups, EMA), IRSDE and the
supervised loss -- for a handful of iterations, without writing into
``experiments/``.

Run from ``codes/config/inpainting``:

    python verify_mtsce_train.py                       # 4 iterations, stage 1
    python verify_mtsce_train.py --iters 10
    python verify_mtsce_train.py -opt options/train/refusion_mask_mtsce_stage2.yml

Checks performed:
  * the pretrained checkpoint loads even though MTSCE is new,
  * with ``finetune.freeze_base: true`` only MTSCE parameters are trainable,
  * MTSCE parameters change while the frozen base stays bit-identical,
  * the loss is finite and the mask reaches the network,
  * the full state dict (base + MTSCE) can be saved and reloaded.
"""

import argparse
import logging
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import options as option  # noqa: E402
import utils as util  # noqa: E402
from data import create_dataloader, create_dataset  # noqa: E402
from models import create_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt",
        default="options/train/refusion_mask_mtsce_stage1.yml",
        help="training option file to smoke-test",
    )
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--crop", type=int, default=64)
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)
    opt = option.dict_to_nonedict(opt)
    opt["dist"] = False

    # keep the smoke run cheap and side-effect free
    train_opt = opt["datasets"]["train"]
    train_opt["batch_size"] = args.batch_size
    train_opt["GT_size"] = args.crop
    train_opt["LR_size"] = args.crop
    train_opt["n_workers"] = 0

    model = create_model(opt)
    net = model.model.module

    print("\n--- model ---")
    print("mtsce enabled      :", model.mtsce_enabled)
    print("insertion points   :", net.mtsce_points)
    counts = net.parameter_count()
    print("mtsce parameters   : {:,}".format(counts["mtsce"]))
    print("total parameters   : {:,}".format(counts["total"]))
    print("freeze base        :", model.freeze_base)

    assert model.mtsce_enabled, "the config under test does not enable MTSCE"
    assert counts["mtsce"] > 0, "no MTSCE parameters were created"

    trainable = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    n_mtsce = len([1 for n, _ in trainable if "mtsce" in n.split(".")])
    print("trainable tensors  : {} (MTSCE: {})".format(len(trainable), n_mtsce))
    if model.freeze_base:
        assert n_mtsce == len(trainable), "freeze_base left non-MTSCE parameters trainable"

    # snapshot a few frozen tensors to prove they do not move
    frozen_snapshot = {
        n: p.detach().clone()
        for n, p in net.named_parameters()
        if "mtsce" not in n.split(".") and n in ("intro.weight", "ending.weight")
    }
    mtsce_snapshot = {
        n: p.detach().clone()
        for n, p in net.named_parameters()
        if "mtsce" in n.split(".")
    }

    train_set = create_dataset(train_opt)
    train_loader = create_dataloader(train_set, train_opt, opt, None)
    sde = util.IRSDE(
        max_sigma=opt["sde"]["max_sigma"],
        T=opt["sde"]["T"],
        schedule=opt["sde"]["schedule"],
        eps=opt["sde"]["eps"],
        device=model.device,
    )
    sde.set_model(model.model)

    print("\n--- {} training iterations ---".format(args.iters))
    losses = []
    mask_seen = False
    step = 0
    for train_data in train_loader:
        step += 1
        GT = train_data["GT"]
        LQ = train_data["LQ"]
        mask = train_data.get("mask", None)
        mask_seen = mask_seen or mask is not None

        timesteps, states = sde.generate_random_states(x0=GT, mu=LQ)
        model.feed_data(states, LQ, GT, mask)
        model.optimize_parameters(step, timesteps, sde)
        model.update_learning_rate(step, warmup_iter=opt["train"]["warmup_iter"])

        loss = model.get_current_log()["loss"]
        losses.append(loss)
        print(
            "  iter {} | loss {:.6e} | lr {:.3e}".format(
                step, loss, model.get_current_learning_rate()
            )
        )
        if step >= args.iters:
            break

    assert len(losses) == args.iters, "did not run the requested number of iterations"
    assert all(l == l and abs(l) != float("inf") for l in losses), "non-finite loss"
    assert mask_seen, "the dataset did not provide a degradation mask"

    moved_mtsce = [
        n
        for n, p in net.named_parameters()
        if n in mtsce_snapshot and not torch.equal(p.detach(), mtsce_snapshot[n])
    ]
    assert moved_mtsce, "MTSCE parameters did not update"
    print("\nMTSCE tensors updated: {}/{}".format(len(moved_mtsce), len(mtsce_snapshot)))
    if len(moved_mtsce) < len(mtsce_snapshot):
        # expected for very short runs: Proj_out is zero-initialised, so the
        # upstream sub-modules only start receiving gradients from step 2 on
        print(
            "  (not updated yet: {})".format(
                sorted(set(mtsce_snapshot) - set(moved_mtsce))[:4]
            )
        )

    if model.freeze_base:
        for name, before in frozen_snapshot.items():
            after = dict(net.named_parameters())[name].detach()
            assert torch.equal(before, after), "{} changed although it is frozen".format(name)
        print("frozen base tensors unchanged: OK")

    # save / reload the full checkpoint (base + MTSCE)
    with tempfile.TemporaryDirectory() as tmp:
        model.opt["path"]["models"] = tmp
        model.save("smoke")
        path = os.path.join(tmp, "smoke_G.pth")
        assert os.path.exists(path), "checkpoint was not written"
        state = torch.load(path, map_location="cpu")
        assert any(k.startswith("mtsce.") for k in state), "checkpoint has no MTSCE tensors"
        from models.modules.DenoisingNAFNet_arch import ConditionalNAFNet

        fresh = ConditionalNAFNet(**opt["network_G"]["setting"])
        report = fresh.load_state_dict(state, strict=True)
        print("checkpoint reload: strict OK, {:,} MTSCE tensors".format(
            len([k for k in state if k.startswith("mtsce.")])
        ))

    print("\nTRAINING SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
