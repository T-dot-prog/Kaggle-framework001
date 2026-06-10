import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from multi_agents.config import MODEL_MAPPINGS
from multi_agents.exceptions import PhaseFailedError
from multi_agents.state import State
from multi_agents.utils import ensure_directories, ensure_kb_directory, setup_logger
from multi_agents.workflow import run_phase

PHASE_ORDER = ["reader", "planner", "data_cleaning", "feature_engineering", "modeling"]


def parse_args():
    parser = argparse.ArgumentParser(description="AutoKaggle Multi-Agent Pipeline")
    parser.add_argument("--competition", type=str, default="bank_churn")
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

    # Find the latest completed phase checkpoint
    latest_idx = -1
    state = None
    for i in range(len(PHASE_ORDER) - 1, -1, -1):
        cp = f"{PHASE_ORDER[i]}_complete"
        loaded = State.load_checkpoint(cp)
        if loaded is not None:
            latest_idx = i
            state = loaded
            logger.info("Resumed from latest checkpoint: '%s'", PHASE_ORDER[i])
            break

    if state is None:
        state = State(
            phase=args.start_phase,
            competition_path=competition_path,
            output_dir=output_dir,
        )
        logger.info("Initialized fresh competition: %s", args.competition)

    # Load df_train once before entering the phase loop
    if state.df_train is None:
        state.load_train_df()

    # Run remaining phases in order
    for phase in PHASE_ORDER[latest_idx + 1:]:
        # Check if phase is skipped
        if phase in state.skipped_phases:
            logger.info("Phase '%s' is already skipped, skipping...", phase)
            state.save_checkpoint(f"{phase}_skipped")
            continue

        logger.info("─" * 50)
        logger.info("Step: %s", phase)
        logger.info("─" * 50)

        if phase == "reader":
            from multi_agents.agents.reader import ReaderAgent
            reader = ReaderAgent()
            state = reader.run(state)
            if not state.competition_summary:
                logger.error("ReaderAgent failed. Aborting.")
                sys.exit(1)

        elif phase == "planner":
            from multi_agents.agents.planner import PlannerAgent
            planner = PlannerAgent()
            state = planner.run(state)
            if not state.execution_plan:
                logger.error("PlannerAgent failed. Aborting.")
                sys.exit(1)

        else:
            try:
                state = run_phase(state, phase)
            except PhaseFailedError as e:
                logger.warning("⚠ Phase '%s' failed: %s", phase, e)
                state.skipped_phases.append(phase)
                state.save_checkpoint(f"{phase}_skipped")
                continue
            state.save_checkpoint(f"{phase}_complete")

    os.makedirs(output_dir, exist_ok=True)
    state_path = os.path.join(output_dir, "latest_state.json")
    with open(state_path, "w") as f:
        json.dump(state.to_dict(), f)
    logger.info("State saved to %s", state_path)

    skipped = state.skipped_phases
    logger.info("=" * 60)
    if skipped:
        logger.info("Pipeline finished with %d skipped phase(s): %s", len(skipped), skipped)
    else:
        logger.info("All phases completed successfully!")
    logger.info("  Competition: %s", args.competition)
    logger.info("  Decisions: %d", len(state.decision_memory))
    logger.info("  Code attempts: %d", len(state.code_attempts))
    logger.info("  Errors: %d", len(state.error_logs))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
