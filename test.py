import argparse
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import yaml

from trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--opt",
        type=str,
        default="config/Kwai_KSVQE_test.yml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "-t",
        "--target_set",
        type=str,
        default="val",
        help="Kept for compatibility with the original test11.py.",
    )
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument(
        "--mode",
        choices=["test", "val"],
        default="test",
        help="test writes output.txt; val computes metrics when labels exist.",
    )

    args = parser.parse_args()

    with open(args.opt, "r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)

    test_load_path = opt.get("test_load_path")
    if test_load_path:
        opt["load_path"] = test_load_path

    if opt.get("load_path") and not os.path.exists(opt["load_path"]):
        raise FileNotFoundError(f"Checkpoint not found: {opt['load_path']}")

    # print(opt)

    trainer = Trainer(args, opt)
    if args.mode == "val":
        trainer.inferece_val()
    else:
        trainer.inferece_test()


if __name__ == "__main__":
    main()
