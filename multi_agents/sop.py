import argparse
import os

from config import MODEL_MAPPINGS
from state import State
from utils import ensure_directories, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="AutoKaggle Multi-Agent Pipeline")
    parser.add_argument("--competition", type=str, default="titanic")
    parser.add_argument("--start_phase", type=int, default=1)
    parser.add_argument("--end_phase", type=int, default=6)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--dest_dir", type=str, default="all_tools")
    parser.add_argument("--model", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    ensure_directories()
    logger = setup_logger("sop")

    competition_path = os.path.join(
        os.getcwd(), "multi_agents", "competition", args.competition
    )
    output_dir = os.path.join(
        os.getcwd(), "multi_agents", "experiments_history",
        args.competition,
        args.model or MODEL_MAPPINGS["planner_model"],
        args.dest_dir,
        str(args.run),
    )

    checkpoint_name = f"{args.competition}_phase_{args.start_phase}"
    state = State.load_checkpoint(checkpoint_name)
    if state is None:
        state = State(
            phase=args.start_phase,
            competition_path=competition_path,
            output_dir=output_dir,
        )
        logger.info("Created new state for %s phase %d", args.competition, args.start_phase)
    else:
        logger.info("Loaded checkpoint for %s", checkpoint_name)

    logger.info(
        "AutoKaggle pipeline started for competition: %s (phase %d to %d)",
        args.competition, args.start_phase, args.end_phase,
    )
    print(f"AutoKaggle pipeline started. Phase {args.start_phase} coming soon.")


if __name__ == "__main__":
    main()
