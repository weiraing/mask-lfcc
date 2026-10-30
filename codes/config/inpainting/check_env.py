"""Pre-flight check for deploying to a new machine.

Run this BEFORE training on a fresh server. It resolves every path exactly the
way train.py / test.py will, so a typo fails here instead of 10 minutes into a
run.

    cd codes/config/inpainting
    python check_env.py -opt=options/train/refusion_mask_correction.yml
    python check_env.py -opt=options/test/refusion_mask_correction.yml

Exit code is non-zero when a hard requirement is missing.
"""

import argparse
import os
import os.path as osp
import sys

_HERE = osp.dirname(osp.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import options as option

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

OK = []
WARN = []
FAIL = []


def ok(msg):
    OK.append(msg)
    print("  [ok]   " + msg)


def warn(msg):
    WARN.append(msg)
    print("  [warn] " + msg)


def fail(msg):
    FAIL.append(msg)
    print("  [FAIL] " + msg)


def section(title):
    print("\n== {} ==".format(title))


def list_images(d):
    if not osp.isdir(d):
        return None
    return sorted(f for f in os.listdir(d) if osp.splitext(f)[1].lower() in IMG_EXT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, required=True)
    parser.add_argument("--is-train", action="store_true",
                        help="parse as a training config (default: infer from the path)")
    args = parser.parse_args()

    is_train = args.is_train or ("/train/" in args.opt.replace("\\", "/")
                                 or args.opt.startswith("options/train"))
    opt = option.dict_to_nonedict(option.parse(args.opt, is_train=is_train))

    # ---------------------------------------------------------------- paths
    section("resolved paths")
    root = opt["path"]["root"]
    print("  cwd              : {}".format(os.getcwd()))
    print("  project root     : {}".format(root))
    print("  experiments root : {}".format(opt["path"].get("experiments_root")))
    if not osp.isdir(root):
        fail("project root does not exist: {}".format(root))
    else:
        ok("project root exists")

    ckpt_keys = ("pretrain_model_G", "pretrain_model_C")
    for key in ckpt_keys:
        p = opt["path"].get(key)
        if not p:
            if key == "pretrain_model_C":
                if is_train:
                    ok("{} is empty -> corrector starts from the identity".format(key))
                else:
                    fail("{} is empty but this is a test config; the run would "
                         "have no corrector".format(key))
            else:
                warn("{} is empty".format(key))
            continue
        resolved = osp.abspath(p)
        if osp.isfile(resolved):
            ok("{} -> {} ({:.1f} MB)".format(key, resolved, osp.getsize(resolved) / 1e6))
        else:
            fail("{} -> {} (NOT FOUND)".format(key, resolved))

    # ------------------------------------------------------------- datasets
    section("datasets")
    for phase, ds in opt["datasets"].items():
        n_gt = list_images(ds.get("dataroot_GT") or "")
        n_lq = list_images(ds.get("dataroot_LQ") or "")
        n_mk = list_images(ds.get("dataroot_mask") or "")
        tag = "{}[{}]".format(phase, ds.get("mode"))

        if n_gt is None:
            fail("{}: dataroot_GT is not a directory: {}".format(tag, ds.get("dataroot_GT")))
            continue
        if n_lq is None:
            fail("{}: dataroot_LQ is not a directory: {}".format(tag, ds.get("dataroot_LQ")))
            continue
        if not n_gt or not n_lq:
            fail("{}: empty image list (GT={}, LQ={})".format(tag, len(n_gt or []), len(n_lq or [])))
            continue

        # The dataset pairs LQ[i] with GT[i] by *sorted index*, so the two name
        # lists have to be identical or the pairs are silently scrambled.
        if n_gt == n_lq:
            ok("{}: {} GT / {} LQ, names identical -> pairing is correct".format(
                tag, len(n_gt), len(n_lq)))
        elif len(n_gt) == len(n_lq):
            bad = sorted(set(n_gt) ^ set(n_lq))[:5]
            fail("{}: same count ({}) but different filenames -> LQ/GT pairs are "
                 "SCRAMBLED. e.g. {}".format(tag, len(n_gt), bad))
        else:
            fail("{}: count mismatch GT={} LQ={}".format(tag, len(n_gt), len(n_lq)))

        if n_mk is None:
            warn("{}: no dataroot_mask (the mask channel would be all zeros)".format(tag))
        else:
            if len(n_mk) != len(n_gt):
                warn("{}: {} masks vs {} GT".format(tag, len(n_mk), len(n_gt)))
            stems_gt = {osp.splitext(f)[0] for f in n_gt}
            missing = [f for f in n_mk
                       if osp.splitext(f)[0] not in stems_gt
                       and osp.splitext(f)[0].replace("_mask", "") not in stems_gt]
            if missing:
                warn("{}: {} mask(s) cannot be paired by stem, e.g. {}".format(
                    tag, len(missing), missing[:3]))
            else:
                ok("{}: {} masks, all pair with a GT by stem".format(tag, len(n_mk)))

    # --------------------------------------------------------------- torch
    section("runtime")
    try:
        import torch
    except ImportError:
        warn("torch is not importable in this interpreter -- skipping the "
             "checkpoint/network check")
        torch = None
    if torch is not None:
        print("  torch {} | cuda available: {}".format(torch.__version__, torch.cuda.is_available()))
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print("  gpu{}: {} {:.1f} GB".format(i, props.name, props.total_memory / 1e9))
        else:
            fail("CUDA is not available")
        gpu_ids = opt.get("gpu_ids") or []
        if gpu_ids and max(gpu_ids) >= torch.cuda.device_count():
            fail("gpu_ids {} but only {} GPU(s) visible".format(
                gpu_ids, torch.cuda.device_count()))

    # ------------------------------------------- checkpoint vs. network spec
    if torch is not None and opt["path"].get("pretrain_model_G"):
        section("checkpoint vs network_G")
        try:
            import models.modules as M

            which = opt["network_G"]["which_model_G"]
            setting = dict(opt["network_G"]["setting"])
            net = getattr(M, which)(**setting)
            expected = set(net.state_dict().keys())
            sd = torch.load(osp.abspath(opt["path"]["pretrain_model_G"]), map_location="cpu")
            got = set(k[7:] if k.startswith("module.") else k for k in sd.keys())
            missing = sorted(expected - got)
            unexpected = sorted(got - expected)
            if not missing and not unexpected:
                ok("pretrain_model_G matches {} exactly ({} tensors)".format(
                    which, len(got)))
            else:
                # a corrector checkpoint legitimately carries extra modules
                allow_extra = [k for k in unexpected if k.split(".")[0] in ("net", "gate")]
                if missing or len(allow_extra) != len(unexpected):
                    fail("{} key mismatch: {} missing, {} unexpected. e.g. {}".format(
                        "pretrain_model_G", len(missing), len(unexpected),
                        (missing + unexpected)[:4]))
                else:
                    ok("checkpoint is a corrector bundle ({} extra tensor(s))".format(
                        len(unexpected)))
        except Exception as exc:  # noqa: BLE001 - report, never crash the check
            warn("could not verify the checkpoint: {}".format(exc))

    # -------------------------------------------------------------- summary
    print("\n{} ok, {} warning(s), {} failure(s)".format(len(OK), len(WARN), len(FAIL)))
    if FAIL:
        print("\nFAILURES:")
        for m in FAIL:
            print("  - " + m)
        return 1
    print("Environment looks usable. Note: run verify_correction.py once too.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
