
import os
import re
import csv
import json
import argparse
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

BASE_DIR = Path(__file__).resolve().parent
QUEUE_FILE = BASE_DIR / "viral_video_queue.csv"
RESULTS_DIR = BASE_DIR / "results"
CLIPS_DIR = BASE_DIR / "clips"
TOKEN_FILE = BASE_DIR / "token.json"

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
PRIVACY_STATUS = os.getenv("YOUTUBE_PRIVACY_STATUS", "private")
CATEGORY_ID = os.getenv("YOUTUBE_CATEGORY_ID", "24")


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


def load_queue():
    if not QUEUE_FILE.exists():
        raise FileNotFoundError(f"Queue file not found: {QUEUE_FILE}")

    with QUEUE_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required = {"video_title", "json_done", "post_done"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"CSV must contain exactly these columns: {required}"
            )

        if set(reader.fieldnames) != required:
            raise ValueError(
                "Unexpected CSV columns. Expected only: "
                "video_title,json_done,post_done"
            )

        rows = list(reader)
        return reader.fieldnames, rows


def save_queue(fieldnames, rows):
    temp_file = QUEUE_FILE.with_suffix(".tmp")

    with temp_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    os.replace(temp_file, QUEUE_FILE)


def save_json(json_file, data):
    temp_file = json_file.with_suffix(".tmp")

    temp_file.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    os.replace(temp_file, json_file)


def load_video_json(video_title):
    video_stem = Path(video_title).stem
    json_file = RESULTS_DIR / f"{video_stem}.json"

    if not json_file.exists():
        raise FileNotFoundError(f"JSON result not found: {json_file}")

    with json_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        moments = data
    elif isinstance(data, dict) and isinstance(data.get("moments"), list):
        moments = data["moments"]
    else:
        raise ValueError(
            f"Invalid JSON structure in {json_file}. "
            "Expected a moments array."
        )

    return json_file, data, moments


def get_clip_path(video_title, moment, index):
    video_stem = Path(video_title).stem
    moment_id = str(moment.get("id") or f"{index:02d}").zfill(2)
    expected = CLIPS_DIR / f"{video_stem}_{moment_id}.mp4"

    if expected.exists():
        return expected

    raise FileNotFoundError(
        f"Clip not found for moment {moment_id}. Expected: {expected}"
    )

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


def upload_video(youtube, clip_path, moment, video_title):
    title = str(moment.get("title") or Path(video_title).stem)
    title = title[:100]

    hook = str(moment.get("hook") or "").strip()
    reason = str(moment.get("reason") or "").strip()

    description_parts = [
        hook,
        reason,
        "#Shorts #ViralShorts"
    ]

    description = "\n\n".join(
        part for part in description_parts if part
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


def process_video(youtube, row):
    video_title = row["video_title"].strip()

    print("\n" + "=" * 65)
    print(f"Processing: {video_title}")
    print("=" * 65)

    json_file, data, moments = load_video_json(video_title)

    if not moments:
        print("No moments found. CSV will not be marked complete.")
        return False

    # Process moments in ascending ID order.
    ordered_moments = sorted(
        enumerate(moments, start=1),
        key=lambda item: get_moment_id(item[1], item[0])
    )

    for index, moment in ordered_moments:
        if moment.get("posted") is True:
            print(
                f"Moment {get_moment_id(moment, index):02d} "
                "already posted. Skipping."
            )
            continue

        moment_id = get_moment_id(moment, index)
        clip_path = get_clip_path(video_title, moment, index)

        print(f"\nMoment: {moment_id:02d}")
        print(f"Title: {moment.get('title', 'Untitled')}")

        # Upload first. Never mark posted before success.
        youtube_video_id = upload_video(
            youtube, clip_path, moment, video_title
        )

        moment["posted"] = True
        moment["youtube_video_id"] = youtube_video_id

        # Save immediately after each successful upload.
        save_json(json_file, data)

        print(f"Moment {moment_id:02d} saved as posted in JSON.")

    all_posted = all(
        moment.get("posted") is True for moment in moments
    )

    if all_posted:
        print(f"\nAll {len(moments)} moments posted for {video_title}.")
        return True

    print(f"\nSome moments are still pending for {video_title}.")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload unposted viral moments to YouTube."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check pending clips without uploading."
    )
    args = parser.parse_args()

    fieldnames, rows = load_queue()

    youtube = None
    if not args.dry_run:
        youtube = load_youtube()

    for row in rows:
        if row["post_done"].strip().lower() == "true":
            continue

        if row["json_done"].strip().lower() != "true":
            print(
                f"\nSkipping {row['video_title']}: JSON is not ready."
            )
            continue

        try:
            if args.dry_run:
                json_file, data, moments = load_video_json(
                    row["video_title"]
                )

                pending = [
                    (i, m) for i, m in enumerate(moments, start=1)
                    if m.get("posted") is not True
                ]

                print(f"\n{row['video_title']}: {len(pending)} pending moments")

                for index, moment in pending:
                    clip = get_clip_path(row["video_title"], moment, index)
                    print(f"  {clip.name}")

                continue

            completed = process_video(youtube, row)

            if completed:
                row["post_done"] = "true"
                save_queue(fieldnames, rows)
                print("CSV updated: post_done=true")

        except Exception as error:
            print(f"\nERROR processing {row['video_title']}: {error}")
            print("This video will remain pending in the CSV.")
            print("Already uploaded moments remain saved in the JSON.")

            # Save other queue changes before continuing.
            save_queue(fieldnames, rows)
            continue

    print("\nQueue processing finished.")


if __name__ == "__main__":
    main()
