
import os
import re
import csv
import io
import json
import argparse
from pathlib import Path
from enum import Enum

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.oauth2.service_account import Credentials as ServiceCredentials

BASE_DIR = Path(__file__).resolve().parent
TOKEN_FILE = BASE_DIR / "token.json"

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]

ENV = os.getenv("ENV", "DEV").upper()
PRIVACY_STATUS = os.getenv("YOUTUBE_PRIVACY_STATUS", "public")
CATEGORY_ID = os.getenv("YOUTUBE_CATEGORY_ID", "24")
MAX_UPLOADS_PER_RUN = 1

class Channel(Enum):
    VIRAL_MOMENT = "viral-moment"
    DOCUMENTARY_MOMENT = "documentary-moment"

CHANNEL_CONFIG = {
    Channel.VIRAL_MOMENT: {
        "queue": BASE_DIR / "viral_moment_queue.csv",
        "results": BASE_DIR / "results" / "viral-moment",
        "clips": BASE_DIR / "clips" / "viral-moment",
    },
    Channel.DOCUMENTARY_MOMENT: {
        "queue": BASE_DIR / "viral_documentary_queue.csv",
        "results": BASE_DIR / "results" / "documentary-moment",
        "clips": BASE_DIR / "clips" / "documentary-moment",
    },
}

GDRIVE_JSON_FOLDER_ID = os.getenv("GDRIVE_JSON_FOLDER_ID")


# ============================================================
# GOOGLE SERVICES
# ============================================================

def load_youtube():
    if not TOKEN_FILE.exists():
        raise FileNotFoundError(f"YouTube token not found: {TOKEN_FILE}")

    credentials = Credentials.from_authorized_user_file(
        str(TOKEN_FILE), SCOPES
    )

    if credentials.expired and credentials.refresh_token:
        print("Refreshing YouTube access token...")
        credentials.refresh(Request())
        TOKEN_FILE.write_text(credentials.to_json(), encoding="utf-8")

    if not credentials.valid:
        raise RuntimeError(
            "YouTube credentials are invalid. Run authorize_youtube.py again."
        )

    return build("youtube", "v3", credentials=credentials)


def load_drive():
    if ENV != "PROD":
        return None

    credentials_json = os.getenv("GDRIVE_CREDENTIALS")

    if credentials_json:
        credentials = ServiceCredentials.from_service_account_info(
            json.loads(credentials_json),
            scopes=DRIVE_SCOPES
        )
    else:
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            raise RuntimeError("Google Drive credentials are not configured.")

        credentials = ServiceCredentials.from_service_account_file(
            credentials_path,
            scopes=DRIVE_SCOPES
        )

    if not GDRIVE_JSON_FOLDER_ID:
        raise RuntimeError("GDRIVE_JSON_FOLDER_ID is missing.")

    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def find_drive_json(service, filename):
    escaped = filename.replace("\\", "\\\\").replace("'", "\\'")
    query = (
        f"'{GDRIVE_JSON_FOLDER_ID}' in parents and "
        f"name = '{escaped}' and trashed = false"
    )

    result = service.files().list(
        q=query,
        fields="files(id,name)",
        pageSize=10
    ).execute()

    files = result.get("files", [])
    return files[0] if files else None


def download_drive_json(service, file_id, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=file_id)

    with destination.open("wb") as file:
        downloader = MediaIoBaseDownload(file, request)
        done = False

        while not done:
            _, done = downloader.next_chunk()

    return destination


def get_json_file(channel, video_title, drive):
    config = CHANNEL_CONFIG[channel]
    filename = f"{Path(video_title).stem}.json"
    local_path = config["results"] / filename

    if local_path.exists():
        return local_path

    if ENV != "PROD":
        raise FileNotFoundError(f"JSON result not found: {local_path}")

    drive_filename = f"{channel.value}__{filename}"
    drive_file = find_drive_json(drive, drive_filename)

    if not drive_file:
        raise FileNotFoundError(f"Drive JSON not found: {drive_filename}")

    print(f"[DRIVE] Downloading {drive_filename}")
    return download_drive_json(drive, drive_file["id"], local_path)


def save_json(channel, json_file, data, drive):
    temp_file = json_file.with_suffix(".tmp")
    temp_file.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    os.replace(temp_file, json_file)

    if ENV != "PROD":
        return

    drive_filename = f"{channel.value}__{json_file.name}"
    drive_file = find_drive_json(drive, drive_filename)

    if not drive_file:
        raise FileNotFoundError(
            f"Cannot update missing Drive JSON: {drive_filename}"
        )

    media = MediaFileUpload(
        str(json_file),
        mimetype="application/json",
        resumable=True
    )

    drive.files().update(
        fileId=drive_file["id"],
        media_body=media
    ).execute()

    print(f"[DRIVE] Updated upload progress: {drive_filename}")


# ============================================================
# QUEUE
# ============================================================

def load_queue(queue_file):
    if not queue_file.exists():
        raise FileNotFoundError(f"Queue file not found: {queue_file}")

    with queue_file.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"video_title", "json_done", "post_done"}

        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"CSV must contain columns: {', '.join(sorted(required))}"
            )

        if set(reader.fieldnames) != required:
            raise ValueError("Unexpected CSV columns in queue.")

        return reader.fieldnames, list(reader)


def save_queue(queue_file, fieldnames, rows):
    temp_file = queue_file.with_suffix(".tmp")

    with temp_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    os.replace(temp_file, queue_file)


# ============================================================
# JSON AND CLIP DISCOVERY
# ============================================================

def load_video_json(channel, video_title, drive):
    json_file = get_json_file(channel, video_title, drive)

    with json_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        moments = data
    elif isinstance(data, dict) and isinstance(data.get("moments"), list):
        moments = data["moments"]
    else:
        raise ValueError(f"Invalid JSON structure: {json_file}")

    return json_file, data, moments


def get_moment_id(moment, index):
    try:
        return int(moment.get("id") or index)
    except (TypeError, ValueError):
        return index


def get_clip_path(channel, video_title, index):
    video_stem = Path(video_title).stem
    expected = CHANNEL_CONFIG[channel]["clips"] / f"{video_stem}_{index:02d}.mp4"

    if expected.exists():
        return expected

    raise FileNotFoundError(f"Clip not found: {expected}")


# ============================================================
# YOUTUBE METADATA
# ============================================================

def clean_tags(*values):
    tags = {"shorts", "viral", "viralshorts"}

    for value in values:
        if not value:
            continue

        if isinstance(value, list):
            for item in value:
                if item:
                    tags.add(str(item).strip())
        else:
            tags.update(
                word.strip("#, ")
                for word in re.findall(r"#?\w+", str(value))
                if word.strip("#, ")
            )

    return list(tags)[:30]


def upload_video(youtube, clip_path, moment):
    title = str(moment.get("title") or "")[:100]
    hook = str(moment.get("hook") or "").strip()
    reason = str(moment.get("reason") or "").strip()

    description = "\n\n".join(
        part for part in [hook, reason, "#Shorts #ViralShorts"] if part
    )

    tags = clean_tags(
        title,
        hook,
        moment.get("highlight_words", [])
    )

    body = {
        "snippet": {
            "title": title,
            "description": description[:5000],
            "tags": tags,
            "categoryId": CATEGORY_ID
        },
        "status": {
            "privacyStatus": PRIVACY_STATUS,
            "selfDeclaredMadeForKids": False
        }
    }

    media = MediaFileUpload(
        str(clip_path),
        mimetype="video/mp4",
        chunksize=8 * 1024 * 1024,
        resumable=True
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media
    )

    print(f"Uploading: {clip_path.name}")
    response = None

    while response is None:
        status, response = request.next_chunk()

        if status:
            print(f"Upload progress: {int(status.progress() * 100)}%")

    video_id = response.get("id")

    if not video_id:
        raise RuntimeError("YouTube did not return a video ID.")

    print(f"Upload successful: https://www.youtube.com/watch?v={video_id}")
    return video_id


# ============================================================
# PROCESS ONE VIDEO
# ============================================================


def process_video(youtube, drive, channel, row):
    video_title = row["video_title"]
    json_path, data, moments = load_video_json(channel, video_title, drive)

    if not moments:
        print(f"⚠️ No moments found: {video_title}")
        return False

    for index, moment in enumerate(moments, start=1):
        if moment.get("posted", False):
            continue

        clip_path = get_clip_path(channel, video_title, index)

        if not clip_path.exists():
            print(f"⚠️ Clip not found: {clip_path}")
            continue

        print(f"🚀 Uploading one clip: {clip_path.name}")

        video_id = upload_video(youtube, clip_path, moment)

        if not video_id:
            print(f"❌ Upload failed: {clip_path.name}")
            return False

        moment["posted"] = True
        moment["youtube_video_id"] = video_id

        # Correct argument order
        save_json(channel, json_path, data, drive)

        print(f"✅ Uploaded: https://www.youtube.com/watch?v={video_id}")
        print("⏹️ One-clip limit reached. Stopping this run.")
        return True

    print(f"✅ No pending clips for: {video_title}")
    return True


# ============================================================
# MAIN
# ============================================================

def main(channel_name=None):
    parser = argparse.ArgumentParser(
        description="Upload generated clips to YouTube."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--channel", choices=[c.name for c in Channel])
    args = parser.parse_args()

    selected_name = channel_name or args.channel or os.getenv("CHANNEL", "VIRAL_MOMENT")
    channel = Channel[selected_name.upper()]
    config = CHANNEL_CONFIG[channel]

    print(f"\nEnvironment: {ENV}")
    print(f"Channel: {channel.value}")
    print(f"Queue: {config['queue']}")
    print(f"Results: {config['results']}")
    print(f"Clips: {config['clips']}")

    config["results"].mkdir(parents=True, exist_ok=True)

    fieldnames, rows = load_queue(config["queue"])

    youtube = None
    drive = None

    if not args.dry_run:
        youtube = load_youtube()
        drive = load_drive()
    elif ENV == "PROD":
        drive = load_drive()

    for row in rows:
        if row["post_done"].strip().lower() == "true":
            continue

        if row["json_done"].strip().lower() != "true":
            print(f"Skipping {row['video_title']}: JSON is not ready.")
            continue

        try:
            if args.dry_run:
                json_file, data, moments = load_video_json(
                    channel, row["video_title"], drive
                )

                pending = [
                    (index, moment)
                    for index, moment in enumerate(moments, start=1)
                    if moment.get("posted") is not True
                ]

                print(f"\n{row['video_title']}: {len(pending)} pending moments")

                for index, moment in pending:
                    clip = get_clip_path(channel, row["video_title"], index)
                    print(f"  {clip.name}")

                continue

            completed = process_video(youtube, drive, channel, row)

            if completed:
                json_file, data, moments = load_video_json(
                    channel, row["video_title"], drive
                )

                all_posted = all(
                    moment.get("posted") is True
                    for moment in moments
                )

                if all_posted:
                    row["post_done"] = "true"
                    print(f"✅ All clips uploaded: {row['video_title']}")
                else:
                    print(f"⏳ More clips pending: {row['video_title']}")

                save_queue(config["queue"], fieldnames, rows)
                print("⏹️ Maximum one upload reached. Ending run.")
                break

        except Exception as error:
            print(f"\nERROR processing {row['video_title']}: {error}")
            print("Already uploaded moments remain saved in JSON.")
            save_queue(config["queue"], fieldnames, rows)

    print(f"\nUpload queue finished: {channel.value}")


if __name__ == "__main__":
    main()
