"""Variable-uncooperative variant of sac_att_sas_cr_unc_uncoop.py.

This keeps the same SAC ATT training style and adds configurable n_uncoop.
"""

import argparse

from bluesky_zoo.sector_cr.sector_cr_sas_uncooperative_variable import (
    SectorCR_ATT_sas_uncoop_variable,
)
from training_loop_uncoop_variable import run_training_loop


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train noisy/uncooperative SAC ATT run with configurable uncooperative count."
    )
    parser.add_argument("--n-agents", type=int, default=20)
    parser.add_argument("--n-uncoop", type=int, default=2)
    parser.add_argument("--num-episodes", type=int, default=10_000)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--max-episode-length", type=int, default=150)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.90)
    parser.add_argument("--buffer-size", type=int, default=int(4e6))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--weights-folder",
        type=str,
        default="/Users/sasha/Documents/Code/pettingzoo_multiuse/sac_unc_cr_att_uncooperative_variable_noise",
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default="",
        help="Optional CSV metrics output. Leave empty to disable.",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="none",
        choices=["none", "human", "rgb_array"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    render_mode = None if args.render_mode == "none" else args.render_mode
    csv_file = args.csv_file or None

    env = SectorCR_ATT_sas_uncoop_variable(
        render_mode=render_mode,
        n_agents=args.n_agents,
        n_uncoop=args.n_uncoop,
    )

    run_training_loop(
        env=env,
        weights_folder=args.weights_folder,
        num_episodes=args.num_episodes,
        train_steps=args.train_steps,
        max_episode_length=args.max_episode_length,
        save_every=args.save_every,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        csv_file=csv_file,
    )


if __name__ == "__main__":
    main()
