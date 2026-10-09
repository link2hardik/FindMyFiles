from eval.scripts.configs import app_configs
from eval.scripts.pipeline import process_files, get_predictions, compute_metrics
from pathlib import Path
import json

EVAL_DIR = Path(__file__).resolve().parents[1]  # eval/
DATA_DIR = EVAL_DIR / "data"  # eval/data
QUESTION_PATH = EVAL_DIR / "data" / "questions.json"

for app in app_configs:
    print("=" * 20, app.name, "=" * 20)
    try:
        process_files(app=app, data_dir=DATA_DIR)

        with open(QUESTION_PATH, "r") as q:
            questions = json.load(q)

        predictions = get_predictions(
            app=app,
            questions=questions,
            k_min=5,
            k_max=20,
            k_interval=2,
            is_k_fixed=False,
        )
        labels_dict = {q["question_id"]: [q["document_id"]] for q in questions}
        metrics = compute_metrics(predictions=predictions, labels_dict=labels_dict)
        print(metrics)

    finally:
        app.close()
        app.clean_directory()
