"""
FraudShieldAI — Pipeline Entrypoint

Run the full end-to-end fraud detection pipeline:
  python main.py --stage all
  python main.py --stage preprocess
  python main.py --stage train
  python main.py --stage evaluate
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fraudshield")


def load_configs():
    pipeline_cfg = yaml.safe_load(Path("configs/pipeline.yaml").read_text())
    model_cfg = yaml.safe_load(Path("configs/models.yaml").read_text())
    return pipeline_cfg, model_cfg


def run_preprocess(pipeline_cfg):
    logger.info("=== Stage: Preprocessing & Feature Engineering ===")
    from src.data.preprocessing import run_preprocessing
    run_preprocessing(pipeline_cfg)


def run_train(pipeline_cfg, model_cfg):
    logger.info("=== Stage: Model Training ===")
    from src.models.trainer import run_training
    run_training(pipeline_cfg, model_cfg)


def run_evaluate(pipeline_cfg, model_cfg):
    logger.info("=== Stage: Evaluation & Reporting ===")
    from src.evaluation.evaluator import run_evaluation
    run_evaluation(pipeline_cfg, model_cfg)


def main():
    parser = argparse.ArgumentParser(description="FraudShieldAI Pipeline")
    parser.add_argument(
        "--stage",
        choices=["preprocess", "train", "evaluate", "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    args = parser.parse_args()

    pipeline_cfg, model_cfg = load_configs()

    # Ensure output directories exist
    for key in ("output_dir", "metrics_dir", "plots_dir", "shap_dir"):
        Path(pipeline_cfg["training"][key]).mkdir(parents=True, exist_ok=True)

    if args.stage in ("preprocess", "all"):
        run_preprocess(pipeline_cfg)

    if args.stage in ("train", "all"):
        run_train(pipeline_cfg, model_cfg)

    if args.stage in ("evaluate", "all"):
        run_evaluate(pipeline_cfg, model_cfg)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
