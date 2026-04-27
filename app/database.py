import pandas as pd
from sqlalchemy import create_engine
from app.config import DATABASE_URL
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import MetaData, Table

engine = create_engine(DATABASE_URL)

# =========================
# SAVE DATA
# =========================
def save_to_db(df):
    try:
        if df is None or df.empty:
            print("⚠️ Data kosong")
            return

        # 🔥 replace NaN → None (penting untuk PostgreSQL)
        df = df.where(df.notnull(), None)

        records = df.to_dict(orient="records")

        with engine.begin() as conn:

            metadata = MetaData()
            asteroids_table = Table(
                "asteroids",
                metadata,
                autoload_with=engine
            )

            stmt = insert(asteroids_table).values(records)

            # 🔥 ANTI DUPLICATE
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["neo_reference_id", "event_date"]
            )

            result = conn.execute(stmt)

            print(f"✅ Inserted {result.rowcount} new rows")

    except Exception as e:
        print("⚠️ Error saat insert:", e)

# =========================
# FETCH DATA
# =========================
def fetch_from_db():
    query = "SELECT * FROM asteroids"
    return pd.read_sql(query, engine)