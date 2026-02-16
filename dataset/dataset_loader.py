"""
Dataset loading and validation.
"""
from pathlib import Path
import pandas as pd
import logging

from exceptions import DataLoadError

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = ["video_id", "transcript"]


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load and validate youtube_videos_export.csv.
    Ensures required columns exist and transcripts are non-empty.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Validated DataFrame
        
    Raises:
        DataLoadError: If file not found or validation fails
    """
    path = Path(csv_path)
    
    if not path.exists():
        raise DataLoadError(f"Dataset not found: {csv_path}")
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise DataLoadError(f"Failed to read CSV: {e}") from e
        
    df["transcript"] = df["transcript"].apply(clean_youtube_xml) 
    # Check required columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise DataLoadError(f"Missing required columns: {missing_columns}")
    
    # Remove rows with empty transcripts
    original_len = len(df)
    df = df.dropna(subset=["transcript"])
    df = df[df["transcript"].str.strip().astype(bool)]
    df = df.reset_index(drop=True)
   
    
    removed = original_len - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} rows with empty transcripts")
    
    logger.info(f"Loaded dataset with {len(df)} videos")
    return df
