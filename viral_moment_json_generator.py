
import csv
import json
import subprocess
import sys
import os
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "viral_video_queue.csv"
VIDEO_DIR = BASE_DIR / "videos"
FINDER_SCRIPT = BASE_DIR / "viral_moments" / "viral_video_finder.py"
RESULTS_DIR = BASE_DIR / "viral_moments" / "results"

def read_queue():
    with open(CSV_FILE, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))

def save_queue(rows):
    fieldnames = ["video_title", "json_done", "post_done"]
    fd, temp_path = tempfile.mkstemp(dir=BASE_DIR, suffix=".csv")
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, CSV_FILE)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def valid_json(json_path):
    if not json_path.exists():
        return False
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return isinstance(data, dict) and isinstance(data.get("moments"), list)
    except (json.JSONDecodeError, OSError):
        return False

def process_queue():
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"Queue CSV not found: {CSV_FILE}")
    if not FINDER_SCRIPT.exists():
        raise FileNotFoundError(f"Finder script not found: {FINDER_SCRIPT}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_queue()

    pending = [
        row for row in rows
        if row.get("json_done", "").strip().lower() != "true"
    ]

    print(f"Total videos: {len(rows)}")
    print(f"Pending JSON generation: {len(pending)}")

    for row in pending:
        video_title = row["video_title"].strip()
        if not video_title:
            continue

        video_path = VIDEO_DIR / video_title
        json_path = RESULTS_DIR / f"{Path(video_title).stem}_clips.json"

        print(f"\n{'=' * 60}")
        print(f"Processing: {video_title}")
        print(f"{'=' * 60}")

        if not video_path.is_file():
            print(f"SKIPPED: Video not found: {video_path}")
            continue

        # Reuse an already generated, valid JSON instead of analyzing again.
        if valid_json(json_path):
            print(f"Valid JSON already exists: {json_path.name}")
            row["json_done"] = "true"
            save_queue(rows)
            continue

        try:
            subprocess.run(
                [
                    sys.executable,
                    str(FINDER_SCRIPT),
                    "--video",
                    video_title
                ],
                cwd=BASE_DIR,
                check=True
            )

            if not valid_json(json_path):
                raise RuntimeError("Finder finished, but its JSON output is missing or invalid.")

            row["json_done"] = "true"
            save_queue(rows)
            print(f"SUCCESS: JSON generated for {video_title}")

        except subprocess.CalledProcessError as error:
            print(f"FAILED: {video_title} (exit code: {error.returncode})")
        except Exception as error:
            print(f"FAILED: {video_title} - {error}")

    completed = sum(
        row.get("json_done", "").strip().lower() == "true"
        for row in rows
    )

    print(f"\nQueue finished. JSON completed: {completed}/{len(rows)}")

if __name__ == "__main__":
    process_queue()
