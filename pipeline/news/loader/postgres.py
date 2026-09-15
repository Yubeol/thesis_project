"""Load original and English news text into the existing news table."""

from pipeline.common.loading import main

if __name__ == "__main__":
    raise SystemExit(main("news"))
