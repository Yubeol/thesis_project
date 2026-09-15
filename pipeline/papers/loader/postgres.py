"""Load reviewed/preprocessed paper records into the existing papers table."""

from pipeline.common.loading import main

if __name__ == "__main__":
    raise SystemExit(main("papers"))
