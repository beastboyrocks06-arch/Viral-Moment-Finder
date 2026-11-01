
import os
import re
import csv
import json
import argparse
import sys
from pathlib import Path
from enum import Enum

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
TOKEN_FILE = BASE_DIR / "token.json"

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]

ENV = os.getenv("ENV", "DEV").upper()
PRIVACY_STATUS = os.getenv("YOUTUBE_PRIVACY_STATUS", "public")
CATEGORY_ID = os.getenv("YOUTUBE_CATEGORY_ID", "24")

MAX_UPLOADS_PER_RUN = 1

GDRIVE_JSON_FOLDER_ID = os.getenv("GDRIVE_JSON_FOLDER_ID")
GDRIVE_CLIPS_FOLDER_ID = os.getenv("GDRIVE_CLIPS_FOLDER_ID")


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


# ============================================================
# GOOGLE SERVICES
# ============================================================

def load_youtube():
    if not TOKEN_FILE.exists():
        raise FileNotFoundError(
            f"YouTube token not found: {TOKEN_FILE}"
        )

    credentials = Credentials.from_authorized_user_file(
        str(TOKEN_FILE),
        SCOPES
    )

    if not credentials.valid:
        if credentials.expired and credentials.refresh_token:
            print("[YOUTUBE] Refreshing OAuth token...")
            credentials.refresh(Request())
            TOKEN_FILE.write_text(
                credentials.to_json(),
                encoding="utf-8"
            )
            print("[YOUTUBE] OAuth token refreshed.")
        else:
            raise RuntimeError(
                "YouTube credentials are invalid or expired. "
                "Run authorize_youtube.py again."
            )

    service = build(
        "youtube",
        "v3",
        credentials=credentials,
        cache_discovery=False
    )

    print("[YOUTUBE] OAuth authentication successful.")
    return service


def load_drive():
    if ENV != "PROD":
        return None

    token_json = os.getenv("GDRIVE_TOKEN_JSON")

    if not token_json:
        raise RuntimeError(
            "GDRIVE_TOKEN_JSON environment variable is missing."
        )

    if not GDRIVE_JSON_FOLDER_ID:
        raise RuntimeError(
            "GDRIVE_JSON_FOLDER_ID environment variable is missing."
        )

    if not GDRIVE_CLIPS_FOLDER_ID:
        raise RuntimeError(
            "GDRIVE_CLIPS_FOLDER_ID environment variable is missing."
        )

    credentials = Credentials.from_authorized_user_info(
        json.loads(token_json),
        scopes=DRIVE_SCOPES
    )

    if not credentials.valid:
        if credentials.expired and credentials.refresh_token:
            print("[DRIVE] Refreshing OAuth token...")
            credentials.refresh(Request())
            print("[DRIVE] OAuth token refreshed.")
        else:
            raise RuntimeError(
                "Google Drive OAuth token is invalid or expired."
            )

    service = build(
        "drive",
        "v3",
        credentials=credentials,
        cache_discovery=False
    )

    account = service.about().get(
        fields="user(displayName,emailAddress)"
    ).execute()

    user = account.get("user", {})

    print(
        f"[DRIVE] Connected: "
        f"{user.get('emailAddress', 'Unknown account')}"
    )

    return service


# ============================================================
# GOOGLE DRIVE FILE OPERATIONS
# ============================================================

def escape_drive_query(value):
    return value.replace("\\", "\\\\").replace("'", "\\'")


def find_drive_file(service, folder_id, filename):
    escaped = escape_drive_query(filename)

    query = (
        f"'{folder_id}' in parents and "
        f"name = '{escaped}' and trashed = false"
    )

    result = service.files().list(
        q=query,
        fields="files(id,name,mimeType,size)",
        pageSize=100
    ).execute()

    files = result.get("files", [])

    if not files:
        return None

    if len(files) > 1:
        print(
            f"[DRIVE] Warning: Multiple files named {filename}. "
            "Using the first match."
        )

    return files[0]


def download_drive_file(service, file_id, destination, label="FILE"):
    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    request = service.files().get_media(
        fileId=file_id
    )

    temp_file = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    try:
        with temp_file.open("wb") as file:
            downloader = MediaIoBaseDownload(
                file,
                request,
                chunksize=8 * 1024 * 1024
            )

            done = False

            while not done:
                status, done = downloader.next_chunk()

                if status:
                    print(
                        f"[DRIVE] {label} download: "
                        f"{int(status.progress() * 100)}%"
                    )

        if temp_file.stat().st_size == 0:
            raise RuntimeError(
                f"Downloaded {label} is empty: {destination.name}"
            )

        os.replace(temp_file, destination)

    except Exception:
        if temp_file.exists():
            temp_file.unlink()
        raise

    print(f"[DRIVE] Downloaded: {destination.name}")
    return destination


def find_drive_json(service, filename):
    return find_drive_file(
        service,
        GDRIVE_JSON_FOLDER_ID,
        filename
    )


def get_json_file(channel, video_title, drive=None):
    filename = f"{Path(video_title).stem}.json"
    json_path = channel.results_dir / filename

    if not json_path.exists():
        raise FileNotFoundError(f"Local JSON not found: {json_path}")

    return json_path

def save_json(json_path, data, drive=None):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[JSON] Updated locally: {json_path}")
    
# ============================================================
# QUEUE OPERATIONS
# ============================================================

def load_queue(queue_file):
    if not queue_file.exists():
        raise FileNotFoundError(
            f"Queue file not found: {queue_file}"
        )

    with queue_file.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        required = {
            "video_title",
            "json_done",
            "post_done"
        }

        if not reader.fieldnames or not required.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                "CSV must contain columns: "
                + ", ".join(sorted(required))
            )

        # Extra columns such as clip_done are allowed and preserved.
        return reader.fieldnames, list(reader)


def save_queue(queue_file, fieldnames, rows):
    temp_file = queue_file.with_suffix(".tmp")

    with temp_file.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(rows)

    os.replace(temp_file, queue_file)

    print(f"[QUEUE] Saved: {queue_file.name}")


def is_true(value):
    return str(value or "").strip().lower() in {
        "true", "1", "yes"
    }


# ============================================================
# JSON AND CLIP DISCOVERY
# ============================================================

def load_video_json(channel, video_title, drive):
    json_file = get_json_file(
        channel,
        video_title,
        drive
    )

    with json_file.open(
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    if isinstance(data, list):
        moments = data
    elif (
        isinstance(data, dict)
        and isinstance(data.get("moments"), list)
    ):
        moments = data["moments"]
    else:
        raise ValueError(
            f"Invalid JSON structure: {json_file}"
        )

    return json_file, data, moments


def get_moment_id(moment, index):
    try:
        return int(moment.get("id") or index)
    except (TypeError, ValueError):
        return index


def get_clip_path(channel, video_title, index, drive):
    video_stem = Path(video_title).stem

    # This must match the clip naming convention used by your generator.
    filename = f"{video_stem}_{index:02d}.mp4"

    expected = (
        CHANNEL_CONFIG[channel]["clips"] / filename
    )

    if expected.exists() and expected.stat().st_size > 0:
        return expected

    if ENV != "PROD" or drive is None:
        raise FileNotFoundError(
            f"Clip not found locally: {expected}"
        )

    drive_file = find_drive_file(
        drive,
        GDRIVE_CLIPS_FOLDER_ID,
        filename
    )

    if not drive_file:
        raise FileNotFoundError(
            f"Clip not found in Google Drive generated-clips folder: "
            f"{filename}"
        )

    print(f"[DRIVE] Downloading MP4: {filename}")

    return download_drive_file(
        drive,
        drive_file["id"],
        expected,
        "MP4"
    )


# ============================================================
# YOUTUBE METADATA
# ============================================================

def clean_tags(*values):
    tags = {
        "shorts",
        "viral",
        "viralshorts"
    }

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
                for word in re.findall(
                    r"#?\w+",
                    str(value)
                )
                if word.strip("#, ")
            )

    return list(tags)[:30]


# ============================================================
# YOUTUBE VIDEO UPLOAD
# ============================================================

def upload_video(youtube, clip_path, moment):
    title = str(
        moment.get("title") or ""
    ).strip()[:100]

    if not title:
        raise ValueError(
            f"Missing YouTube title for {clip_path.name}"
        )

    hook = str(moment.get("hook") or "").strip()
    reason = str(moment.get("reason") or "").strip()

    description = "\n\n".join(
        part for part in [
            hook,
            reason,
            "#Shorts #ViralShorts"
        ] if part
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

    print(f"\n[YOUTUBE] Uploading: {clip_path.name}")

    response = None

    while response is None:
        status, response = request.next_chunk()

        if status:
            print(
                f"[YOUTUBE] Upload progress: "
                f"{int(status.progress() * 100)}%"
            )

    video_id = response.get("id")

    if not video_id:
        raise RuntimeError(
            "YouTube did not return a video ID."
        )

    print(
        f"[YOUTUBE] Upload successful: "
        f"https://www.youtube.com/watch?v={video_id}"
    )

    return video_id


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(youtube, drive, channel, row):
    video_title = row["video_title"]

    json_path, data, moments = load_video_json(
        channel,
        video_title,
        drive
    )

    if not moments:
        print(f"[WARNING] No moments found: {video_title}")
        return "empty"

    for index, moment in enumerate(moments, start=1):
        if moment.get("posted") is True:
            continue

        clip_path = get_clip_path(
            channel,
            video_title,
            index,
            drive
        )

        print(f"\n[UPLOAD] Processing: {clip_path.name}")

        video_id = upload_video(
            youtube,
            clip_path,
            moment
        )

        if not video_id:
            raise RuntimeError(
                f"YouTube upload failed: {clip_path.name}"
            )

        moment["posted"] = True
        moment["youtube_video_id"] = video_id

        # Persist upload status to Drive immediately.
        save_json(
            channel,
            json_path,
            data,
            drive
        )

        print(
            f"[SUCCESS] Uploaded: "
            f"https://www.youtube.com/watch?v={video_id}"
        )

        return "uploaded"

    print(f"[SUCCESS] No pending clips for: {video_title}")
    return "complete"


# ============================================================
# MAIN
# ============================================================

def main(channel_name=None):
    parser = argparse.ArgumentParser(
        description="Upload generated clips to YouTube."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check pending clips without uploading."
    )

    parser.add_argument(
        "--channel",
        choices=[channel.name for channel in Channel],
        help="Select channel."
    )

    args = parser.parse_args()

    selected_name = (
        channel_name
        or args.channel
        or os.getenv("CHANNEL", "VIRAL_MOMENT")
    )

    try:
        channel = Channel[selected_name.upper()]
    except KeyError:
        raise ValueError(f"Invalid channel: {selected_name}")

    config = CHANNEL_CONFIG[channel]

    print("\n" + "=" * 60)
    print("YOUTUBE CLIP UPLOADER")
    print("=" * 60)
    print(f"Environment : {ENV}")
    print(f"Channel     : {channel.value}")
    print(f"Privacy     : {PRIVACY_STATUS}")
    print(f"Queue       : {config['queue']}")
    print(f"Results     : {config['results']}")
    print(f"Clips       : {config['clips']}")
    print(f"Dry Run     : {args.dry_run}")
    print("=" * 60)

    config["results"].mkdir(parents=True, exist_ok=True)
    config["clips"].mkdir(parents=True, exist_ok=True)

    fieldnames, rows = load_queue(config["queue"])

    youtube = None
    drive = None

    if not args.dry_run:
        youtube = load_youtube()
        drive = load_drive()
    elif ENV == "PROD":
        drive = load_drive()

    uploads_completed = 0
    had_errors = False

    for row in rows:
        video_title = row["video_title"]

        if is_true(row.get("post_done")):
            continue

        if not is_true(row.get("json_done")):
            print(f"[SKIP] JSON is not ready: {video_title}")
            continue

        # If clip_done exists, only process generated clips.
        if (
            "clip_done" in row
            and not is_true(row.get("clip_done"))
        ):
            print(f"[SKIP] Clips are not ready: {video_title}")
            continue

        try:
            json_file, data, moments = load_video_json(
                channel,
                video_title,
                drive
            )

            pending = [
                (index, moment)
                for index, moment in enumerate(
                    moments,
                    start=1
                )
                if moment.get("posted") is not True
            ]

            if args.dry_run:
                print(
                    f"\n[DRY RUN] {video_title}: "
                    f"{len(pending)} pending moments"
                )

                for index, moment in pending:
                    clip = get_clip_path(
                        channel,
                        video_title,
                        index,
                        drive
                    )
                    print(f"  - {clip.name}")

                continue

            if not pending:
                row["post_done"] = "true"
                save_queue(config["queue"], fieldnames, rows)
                print(f"[SUCCESS] All moments already posted: {video_title}")
                continue

            if uploads_completed >= MAX_UPLOADS_PER_RUN:
                print("[INFO] Maximum upload limit reached.")
                break

            result = process_video(
                youtube,
                drive,
                channel,
                row
            )

            # Reload JSON to verify the saved posting state.
            json_file, data, moments = load_video_json(
                channel,
                video_title,
                drive
            )

            all_posted = (
                bool(moments)
                and all(
                    moment.get("posted") is True
                    for moment in moments
                )
            )

            if all_posted:
                row["post_done"] = "true"
                print(f"[SUCCESS] All clips uploaded: {video_title}")
            else:
                print(f"[PENDING] More clips remaining: {video_title}")

            save_queue(config["queue"], fieldnames, rows)

            if result == "uploaded":
                uploads_completed += 1

            if uploads_completed >= MAX_UPLOADS_PER_RUN:
                print("[INFO] Maximum one-upload limit reached.")
                break

        except Exception as error:
            had_errors = True

            print(
                f"\n[ERROR] Processing {video_title}: {error}"
            )

            print(
                "[WARNING] Check YouTube and Drive status "
                "before retrying."
            )

            save_queue(config["queue"], fieldnames, rows)
            break

    print("\n" + "=" * 60)
    print(f"UPLOAD SUMMARY: {channel.value}")
    print(f"Successful uploads this run: {uploads_completed}")
    print(f"Errors encountered: {had_errors}")
    print("=" * 60)

    if had_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
