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
    VIRAL_REVELATION = "viral-revelation"

CHANNEL = Channel[os.getenv("CHANNEL", "VIRAL_MOMENT").upper()]

CHANNEL_CONFIG = {
    Channel.VIRAL_MOMENT: {
        "finder": BASE_DIR / "viral_moments" / "viral_moment_finder.py",
        "queue": BASE_DIR / "viral_moment_queue.csv"
    },
    Channel.DOCUMENTARY_MOMENT: {
        "finder": BASE_DIR / "viral_documentary" / "documentary_video_finder.py",
        "queue": BASE_DIR / "viral_documentary_queue.csv"
    },
    Channel.VIRAL_REVELATION: {
        "finder": BASE_DIR / "viral_revelation" / "viral_revelation_finder.py",
        "queue": BASE_DIR / "viral_revelation_queue.csv"
    }
}

CSV_FILE = CHANNEL_CONFIG[CHANNEL]["queue"]
FINDER_SCRIPT = CHANNEL_CONFIG[CHANNEL]["finder"]
VIDEO_DIR = BASE_DIR / "videos"
RESULTS_DIR = BASE_DIR / "results"
CHANNEL_RESULTS = RESULTS_DIR / CHANNEL.value
SCOPES = ["https://www.googleapis.com/auth/drive"]
MAX_VIDEOS_PER_RUN = 1

def get_drive():
    credentials_json = os.getenv("GDRIVE_CREDENTIALS", "").strip()
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()

    if credentials_json:
        print("[DRIVE] Using service account credentials from environment.")
        credentials = Credentials.from_service_account_info(
            json.loads(credentials_json),
            scopes=SCOPES
        )
    elif credentials_path:
        print(f"[DRIVE] Using service account file: {credentials_path}")
        credentials = Credentials.from_service_account_file(
            credentials_path,
            scopes=SCOPES
        )
    else:
        raise RuntimeError(
            "Google Drive credentials missing. Configure GDRIVE_CREDENTIALS "
            "or GOOGLE_APPLICATION_CREDENTIALS."
        )

    return build("drive", "v3", credentials=credentials, cache_discovery=False)

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
    print(f"[DRIVE] Searching for: {filename}", flush=True)

    result = service.files().list(
        q=f"'{folder_id}' in parents and name='{filename}' and trashed=false",
        fields="files(id,name,size)",
        pageSize=10,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()

    files = result.get("files", [])
    print(f"[DRIVE] Search completed. Matches: {len(files)}", flush=True)
    return files

def download_drive_file(service, file_id, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DRIVE] Starting download: {destination.name}", flush=True)

    request = service.files().get_media(
        fileId=file_id,
        supportsAllDrives=True
    )

    with open(destination, "wb") as file:
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(
                    f"[DRIVE] Download progress: {int(status.progress() * 100)}%",
                    flush=True
                )

    print(f"[DRIVE] Download complete: {destination}", flush=True)

def upload_drive_file(service, path, folder_id, filename):
    matches = find_drive_file(service, folder_id, filename)
    media = MediaFileUpload(str(path), mimetype="application/json", resumable=True)

    if matches:
        result = service.files().update(
            fileId=matches[0]["id"],
            media_body=media,
            supportsAllDrives=True,
            fields="id,name"
        ).execute()
    else:
        result = service.files().create(
            body={"name": filename, "parents": [folder_id]},
            media_body=media,
            supportsAllDrives=True,
            fields="id,name"
        ).execute()

    print(f"[DRIVE] Uploaded: {result['name']}")
    return result

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

    for row in pending[:MAX_VIDEOS_PER_RUN]:
        video_title = row.get("video_title", "").strip()
        if not video_title:
            continue

        print(f"\n[QUEUE] Starting video: {video_title}", flush=True)
        video_path = get_video(video_title, service)  
        json_path = CHANNEL_RESULTS / f"{Path(video_title).stem}.json"

        print(f"\n{'=' * 50}\nProcessing: {video_title}\n{'=' * 50}")

        if not video_path:
            print(f"SKIPPED: Video not found: {video_title}")
            continue

        try:
            if valid_json(json_path):
                print(f"[JSON] Valid local JSON already exists: {json_path.name}")
            else:
                subprocess.run(
                    [sys.executable, str(FINDER_SCRIPT), "--video", video_title],
                    cwd=BASE_DIR,
                    check=True
                )
                if not valid_json(json_path):
                    raise RuntimeError(f"JSON missing or invalid: {json_path}")

            print(f"[JSON] Saved locally: {json_path}")

            # Local JSON generation is the primary success condition.
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