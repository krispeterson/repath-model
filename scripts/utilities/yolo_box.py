#!/usr/bin/env python3
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repath_model.annotation.yolo_box import main

if __name__ == "__main__":
    main()
