import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

for file in DATA_DIR.glob("*.csv"):
    df = pd.read_csv(file)

    print("=" * 80)
    print(file.name)
    print("=" * 80)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print()