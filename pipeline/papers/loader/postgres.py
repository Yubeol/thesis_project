"""Load reviewed/preprocessed paper records into the existing papers table."""

from pathlib import Path

from pipeline.common.loading import main

if __name__ == "__main__":
    raise SystemExit(main("papers", Path("data/processed/papers/papers_ready.json")))
