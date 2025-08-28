from venv import logger
from autoemulate.core.results import Result  # , Results
from pathlib import Path
from autoemulate.emulators.base import Emulator
import pandas as pd
import ast
import joblib

def ae_load_result(path: str | Path) -> Result | Emulator:
    """
    Load a result or model from disk.

    Load a model and (if it exists) its metadata from disk,
    returning either a Result or Emulator object.

    Parameters
    ----------
    path : str or Path
        Path to the model file.

    Returns
    -------
    Result or Emulator
        The loaded model or result object.
    """
    if ".joblib" not in str(path):
        model_path = Path(f"{path}.joblib")
    else:
        model_path = Path(path)
    model = ae_load_model(model_path)
    metadata_path = Path(f"{path}_metadata.csv")
    try:
        metadata_df = pd.read_csv(metadata_path, nrows=1)
        logger.info("Metadata loaded from %s", metadata_path)
    except Exception as e:
        msg = "Failed to load metadata from %s: %s"
        logger.error(msg, metadata_path, e)
        return model
    row = metadata_df.iloc[0]
    params = row["params"]
    params = ast.literal_eval(params)

    return Result(
        id=row["id"],
        model_name=row["model_name"],
        model=model,
        params=params,
        r2_test=row["r2_test"],
        rmse_test=row["rmse_test"],
        r2_test_std=row["r2_test_std"],
        rmse_test_std=row["rmse_test_std"],
        r2_train=row["r2_train"],
        rmse_train=row["rmse_train"],
        r2_train_std=row["r2_train_std"],
        rmse_train_std=row["rmse_train_std"],
    )

def ae_load_model(path: str | Path):
    """Load a model from disk.

    Parameters
    ----------
    path : str or Path
        Path to load the model.
    """
    path = Path(path)
    try:
        model = joblib.load(path)
        logger.info("Model loaded from %s", path)
        return model
    except Exception as e:
        logger.error("Failed to load model from %s: %s", path, e)
        raise

def swap_s1_st(sobol_df):

    # Due to a bug in autoemulate plotting, we must swap ST and S1 rows.

    # Get the indices of rows where index == 'ST'
    mask_st = sobol_df['index'] == 'ST'
    mask_s1 = sobol_df['index'] == 'S1'

    # For those rows, change the index to be 'S1'
    sobol_df.loc[mask_st, 'index'] = 'S1'

    # For those rows, change the index to be 'ST'
    sobol_df.loc[mask_s1, 'index'] = 'ST'

    return sobol_df
