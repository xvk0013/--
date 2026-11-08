"""Move MB recordings 21 and 33 to Train and exchange 50 and 27 into Val."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shipsear_val_background_20261104"))
from prepare_pf_swap import BANK, build_dataset


if __name__ == "__main__":
    build_dataset(
        BANK / "datasets/shipsear_pf_mb_rv_0db",
        HERE / "data",
        {"ShipsEar_21": "ShipsEar_50", "ShipsEar_50": "ShipsEar_21",
         "ShipsEar_33": "ShipsEar_27", "ShipsEar_27": "ShipsEar_33"},
        {"ShipsEar_21": "Train", "ShipsEar_33": "Train", "ShipsEar_27": "Val", "ShipsEar_50": "Val"},
        "MB",
        "shipsear_mb21_33_train_27_50_val_20261104",
        "Difficulty-guided Train/Val MB exchange. Test retained from the original split.",
    )
