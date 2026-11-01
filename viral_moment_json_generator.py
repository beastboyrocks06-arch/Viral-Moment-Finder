import csv, json, subprocess, sys, os, tempfile
from pathlib import Path
from enum import Enum
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import io

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
ENV = os.getenv("ENV", "DEV").upper()

class Channel(Enum):
    VIRAL_MOMENT = "viral-moment"
    DOCUMENTARY_MOMENT = "documentary-moment"

CHANNEL = Channel[os.getenv("CHANNEL", "VIRAL_MOMENT").upper()]

CHANNEL_CONFIG = {
    Channel.VIRAL_MOMENT: {
        "finder": BASE_DIR / "viral_moments" / "viral_video_finder.py",
        "queue": BASE_DIR / "viral_moment_queue.csv"
    },
    Channel.DOCUMENTARY_MOMENT: {
        "finder": BASE_DIR / "documentary_finder" / "documentary_video_finder.py",
        "queue": BASE_DIR / "viral_documentary_queue.csv"
    }
}

CSV_FILE = CHANNEL_CONFIG[CHANNEL]["queue"]
FINDER_SCRIPT = CHANNEL_CONFIG[CHANNEL]["finder"]
VIDEO_DIR = BASE_DIR / "videos"
RESULTS_DIR = BASE_DIR / "results"
CHANNEL_RESULTS = RESULTS_DIR / CHANNEL.value
SCOPES = ["https://www.googleapis.com/auth/drive"]

def get_drive():
    credentials_json = os.getenv("GDRIVE_CREDENTIALS")
    if credentials_json:
        credentials = Credentials.from_service_account_info(json.loads(credentials_json), scopes=SCOPES)
    else:
        credentials = Credentials.from_service_account_file(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"), scopes=SCOPES)
    return build("drive", "v3", credentials=credentials)

def read_queue():
    with open(CSV_FILE, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))

def save_queue(rows):
    fd, temp_path = tempfile.mkstemp(dir=BASE_DIR, suffix=".csv")
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["video_title", "json_done", "post_done"])
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, CSV_FILE)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def valid_json(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return isinstance(data, dict) and isinstance(data.get("moments"), list)
    except (json.JSONDecodeError, OSError, TypeError):
        return False

def find_drive_file(service, folder_id, filename):
    result = service.files().list(q=f"'{folder_id}' in parents and name='{filename}' and trashed=false", fields="files(id,name)", pageSize=10).execute()
    return result.get("files", [])

def download_drive_file(service, file_id, destination):
    request = service.files().get_media(fileId=file_id)
    with open(destination, "wb") as file:
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

def upload_drive_file(service, path, folder_id, filename):
    matches = find_drive_file(service, folder_id, filename)
    media = MediaFileUpload(str(path), mimetype="application/json", resumable=True)
    if matches:
        service.files().update(fileId=matches[0]["id"], media_body=media).execute()
    else:
        service.files().create(body={"name": filename, "parents": [folder_id]}, media_body=media, fields="id").execute()

def get_video(video_title, service=None):
    local_path = VIDEO_DIR / video_title
    if ENV == "DEV":
        return local_path if local_path.is_file() else None
    folder_id = os.getenv("GDRIVE_SOURCE_FOLDER_ID")
    if not folder_id:
        raise ValueError("GDRIVE_SOURCE_FOLDER_ID is missing")
    matches = find_drive_file(service, folder_id, video_title)
    if not matches:
        return None
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    download_drive_file(service, matches[0]["id"], local_path)
    return local_path

def process_queue():
    if ENV not in ("DEV", "PROD"):
        raise ValueError("ENV must be DEV or PROD")
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"Queue CSV not found: {CSV_FILE}")
    if not FINDER_SCRIPT.is_file():
        raise FileNotFoundError(f"Finder script not found: {FINDER_SCRIPT}")

    service = get_drive() if ENV == "PROD" else None
    CHANNEL_RESULTS.mkdir(parents=True, exist_ok=True)
    rows = read_queue()
    pending = [row for row in rows if row.get("json_done", "").strip().lower() != "true"]
    print(f"Environment: {ENV} | Channel: {CHANNEL.value}")
    print(f"Total videos: {len(rows)} | Pending: {len(pending)}")

    for row in pending:
        video_title = row.get("video_title", "").strip()
        if not video_title:
            continue

        video_path = get_video(video_title, service)
        json_path = CHANNEL_RESULTS / f"{Path(video_title).stem}.json"
        drive_name = f"{CHANNEL.value}__{json_path.name}"

        print(f"\n{'=' * 50}\nProcessing: {video_title}\n{'=' * 50}")

        if not video_path:
            print(f"SKIPPED: Video not found: {video_title}")
            continue

        try:
            if ENV == "PROD":
                folder_id = os.getenv("GDRIVE_JSON_FOLDER_ID")
                if not folder_id:
                    raise ValueError("GDRIVE_JSON_FOLDER_ID is missing")
                existing = find_drive_file(service, folder_id, drive_name)
                if existing:
                    download_drive_file(service, existing[0]["id"], json_path)

            if valid_json(json_path):
                print(f"Valid JSON already exists: {json_path.name}")
            else:
                subprocess.run([sys.executable, str(FINDER_SCRIPT), "--video", video_title], cwd=BASE_DIR, check=True)
                if not valid_json(json_path):
                    raise RuntimeError(f"JSON missing or invalid: {json_path}")

            if ENV == "PROD":
                upload_drive_file(service, json_path, os.getenv("GDRIVE_JSON_FOLDER_ID"), drive_name)

            row["json_done"] = "true"
            save_queue(rows)
            print(f"SUCCESS: {video_title}")

        except subprocess.CalledProcessError as error:
            print(f"FAILED: {video_title} (exit code: {error.returncode})")
        except Exception as error:
            print(f"FAILED: {video_title} - {error}")

    completed = sum(row.get("json_done", "").strip().lower() == "true" for row in rows)
    print(f"\nQueue finished. JSON completed: {completed}/{len(rows)}")

if __name__ == "__main__":
    process_queue()