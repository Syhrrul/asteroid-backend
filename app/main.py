from fastapi import FastAPI, BackgroundTasks
from app.pipeline import run_pipeline
from app.database import fetch_from_db

app = FastAPI()

pipeline_status = {"running": False}

@app.get("/")
def root():
    return {"message": "Asteroid API running 🚀"}

# =========================
# GET DATA
# =========================
@app.get("/asteroids")
def get_asteroids():
    df = fetch_from_db()
    return df.to_dict(orient="records")

# =========================
# RUN PIPELINE MANUAL
# =========================
@app.post("/run-pipeline")
def trigger_pipeline(background_tasks: BackgroundTasks):
    if pipeline_status["running"]:
        return {"status": "pipeline already running"}

    pipeline_status["running"] = True

    def task():
        try:
            run_pipeline()
        finally:
            pipeline_status["running"] = False

    background_tasks.add_task(task)

    return {"status": "pipeline started"}
# =========================
# CHECK NAN ROWS
# =========================
@app.get("/nan-rows")
def get_nan_rows():
    df = fetch_from_db()
    df_nan = df[df.isna().any(axis=1)]
    return df_nan.to_dict(orient="records")