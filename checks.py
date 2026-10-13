import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def check_missing(df):
    """Percentage of missing values per column (only columns that have some)."""
    missing = df.isna().mean() * 100
    return missing[missing > 0].round(2).sort_values(ascending=False)


def check_duplicates(df):
    """Number of fully duplicated rows."""
    return int(df.duplicated().sum())


def check_constant_columns(df):
    """Columns that have the same value in every row (useless for ML)."""
    return [col for col in df.columns if df[col].nunique(dropna=False) <= 1]


def run_all(path):
    df = load_data(path)
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}\n")

    print("Missing values (%):")
    missing = check_missing(df)
    print(missing.to_string() if not missing.empty else "None found")

    print(f"\nDuplicate rows: {check_duplicates(df)}")

    constant = check_constant_columns(df)
    print(f"Constant columns: {constant if constant else 'None found'}")


if __name__ == "__main__":
    run_all("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")