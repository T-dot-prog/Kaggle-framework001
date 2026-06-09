import argparse
import os
import sys

from config import MODEL_MAPPINGS
from state import State
from utils import ensure_directories, ensure_kb_directory, setup_logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


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
    ensure_kb_directory()
    logger = setup_logger("sop")

    competition_path = os.path.join(
        os.getcwd(), "multi_agents", "competition", args.competition
    )
    output_dir = os.path.join(
        os.getcwd(), "multi_agents", "experiments_history",
        args.competition,
        str(args.run),
    )

    planner_checkpoint = "planner_complete"
    reader_checkpoint = "reader_complete"

    state = State.load_checkpoint(planner_checkpoint)
    if state is not None:
        logger.info("Planner checkpoint found — skipping Reader and Planner")
    else:
        state = State.load_checkpoint(reader_checkpoint)
        if state is not None:
            logger.info("Reader checkpoint found — skipping Reader")
        else:
            state = State(
                phase=args.start_phase,
                competition_path=competition_path,
                output_dir=output_dir,
            )
            logger.info("Created new state for %s phase %d", args.competition, args.start_phase)

            logger.info("Initialized fresh competition: %s", args.competition)
            logger.info("─" * 50)
            logger.info("Step 1/2: ReaderAgent – analyzing competition data")
            logger.info("─" * 50)

            from agents.reader import ReaderAgent
            reader = ReaderAgent()
            state = reader.run(state)

            if not state.competition_summary:
                logger.error("ReaderAgent failed to produce summary. Aborting.")
                sys.exit(1)

            logger.info("ReaderAgent complete.")

        logger.info("─" * 50)
        logger.info("Step 2/2: PlannerAgent – generating execution plan")
        logger.info("─" * 50)

        from agents.planner import PlannerAgent
        planner = PlannerAgent()
        state = planner.run(state)

        if not state.execution_plan:
            logger.error("PlannerAgent failed to produce a plan. Aborting.")
            sys.exit(1)

        logger.info("PlannerAgent complete.")

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    import json
    with open(os.path.join(os.path.dirname(output_dir), "latest_state.json"), "w") as f:
        json.dump(state.to_dict(), f)
    logger.info("State saved.")

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("  Competition: %s", args.competition)
    logger.info("  Decisions: %d", len(state.decision_memory))
    logger.info("  Errors: %d", len(state.error_logs))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
