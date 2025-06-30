# check_zip_contents_utils.py
import zipfile
import io
import pickle
import pandas as pd


def check_zip_contents(zip_file) -> tuple[bool, dict | None, str | None]:

    required_files = {"train_data.csv", "test_data.csv", "scalers.pkl"}

    try:
        with zipfile.ZipFile(zip_file) as z:
            zip_contents = set(z.namelist())

            # Check for required files
            if not required_files.issubset(zip_contents):
                missing = required_files - zip_contents
                return False, None, f"Missing required files in ZIP: {', '.join(missing)}"

            # Read CSVs into pandas DataFrames
            with z.open("train_data.csv") as train_file:
                train_df = pd.read_csv(train_file)

            with z.open("test_data.csv") as test_file:
                test_df = pd.read_csv(test_file)

            # Load scalers from pickle file
            with z.open("scalers.pkl") as scalers_file:
                scalers = pickle.load(scalers_file)

            return True, {
                "train_df": train_df,
                "test_df": test_df,
                "scalers": scalers
            }, None

    except zipfile.BadZipFile:
        return False, None, "Uploaded file is not a valid ZIP file."
    except Exception as e:
        return False, None, f"Error while reading ZIP file: {str(e)}"
