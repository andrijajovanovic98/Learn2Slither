from dataclasses import dataclass


@dataclass
class Config:
    board_size: int
    episodes: int
    visual: bool
    step: bool
    dontlearn: bool
    save_path: str | None
    load_path: str | None
    fps: float
    assist: bool


def make_config(args) -> Config:
    return Config(
        board_size=args.board,
        episodes=args.episodes,
        visual=args.visual,
        step=args.step,
        dontlearn=args.dontlearn,
        save_path=args.save,
        load_path=args.load,
        fps=args.speed,
        assist=not args.dontlearn
    )
