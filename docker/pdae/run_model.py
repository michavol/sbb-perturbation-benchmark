#!/usr/bin/env python3
"""
Main entry point for pDAE Docker container.
Supports both training and prediction modes.
"""

import json
import logging
import sys

from pdae_wrapper import PDAEWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python run_model.py {train|predict} /path/to/config.json")
        sys.exit(1)

    mode = sys.argv[1]
    config_path = sys.argv[2]

    if mode not in ["train", "predict"]:
        log.error("Unknown mode: %s. Must be 'train' or 'predict'", mode)
        sys.exit(1)

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception as exc:
        log.error("Failed to load config from %s: %s", config_path, exc)
        sys.exit(1)

    if config.get("mode") != mode:
        log.error(
            "Config mode '%s' does not match command mode '%s'",
            config.get("mode"),
            mode,
        )
        sys.exit(1)

    try:
        wrapper = PDAEWrapper(config)
        if mode == "train":
            wrapper.train()
        else:
            wrapper.predict()
    except Exception as exc:
        log.exception("pDAE %s failed: %s", mode, exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
