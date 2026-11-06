import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
DB_PATH = DATA_DIR / "instamart.db"

def main():
    required_files = {
        "customers": CLEAN_DIR / "customers.csv",
        "products": CLEAN_DIR / "products.csv",
        "orders": CLEAN_DIR / "orders.csv",
    }

    for table, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    with sqlite3.connect(DB_PATH) as conn:
        for table, path in required_files.items():
            df = pd.read_csv(path)
            df.to_sql(table, conn, if_exists="replace", index=False)
            print(f"{table}: {len(df)} rows imported")

        tables = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' ORDER BY name"
        ).fetchall()

        print("Tables:", [row[0] for row in tables])

    print(f"Database created: {DB_PATH}")

if __name__ == "__main__":
    main()
