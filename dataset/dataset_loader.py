from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = ["video_id", "transcript"]


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load and validate youtube_videos_export.csv.
    Ensures required columns exist and transcripts are non-empty.
    """

    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(path)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.dropna(subset=["transcript"])
    df = df[df["transcript"].str.strip().astype(bool)]
    df = df.reset_index(drop=True)

    return df
