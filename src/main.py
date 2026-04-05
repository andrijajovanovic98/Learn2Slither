import argparse
from pathlib import Path
from utils.config import make_config
from train.trainer import Trainer
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "agent"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_DIR / "agent_actions.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--visual", action="store_true", help="Graphical display ON")
    p.add_argument("--step", action="store_true", help="Step-by-step mode")
    p.add_argument("--dontlearn", action="store_true", help="Disable learning (eval)")
    p.add_argument("--save", type=str, default=None, help="Path to save model")
    p.add_argument("--load", type=str, default=None, help="Path to load model")
    p.add_argument("--speed", type=float, default=6.0, help="FPS (human readable >=~2)")
    p.add_argument("--board", type=int, default=10, help="Board size NxN (bonus)")

    return p.parse_args()


def main():
    args = parse_args()
    cfg = make_config(args)
    if cfg.load_path and not Path(cfg.load_path).is_absolute():
        cfg.load_path = str(PROJECT_ROOT / cfg.load_path)
    if cfg.save_path and not Path(cfg.save_path).is_absolute():
        cfg.save_path = str(PROJECT_ROOT / cfg.save_path)
    tr = Trainer(cfg)
    tr.run()


if __name__ == "__main__":
    main()
