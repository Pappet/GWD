"""Find models in the model registry folder.

Start of the registry (P2.5): get_latest_model moved here unchanged from the leaderboard
module. P3.3 makes this the one model search and changes how the latest model is chosen.
"""

import glob
import os
from typing import Optional


def get_latest_model(models_dir: str) -> Optional[str]:
    """Sucht das neueste Modell im Registry-Ordner."""
    if not os.path.exists(models_dir):
        print(f"❌ Ordner '{models_dir}' nicht gefunden.")
        return None

    files = glob.glob(os.path.join(models_dir, "*.keras"))
    if not files:
        print("❌ Keine Modelle (.keras) gefunden.")
        return None

    latest_file = max(files, key=os.path.getctime)
    return latest_file
