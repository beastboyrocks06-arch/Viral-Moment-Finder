
import argparse
import csv
import json
import math
import os
import random
import tempfile
import traceback
import sys
from pathlib import Path
from enum import Enum

import numpy as np
from PIL import Image
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

from moviepy import (
    VideoFileClip,
    ImageClip,
    CompositeVideoClip,
    CompositeAudioClip,
    concatenate_videoclips
)
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.fx import FadeIn


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import caption_generator as caption


class Channel(Enum):
    VIRAL_MOMENT = "viral-moment"
    DOCUMENTARY_MOMENT = "documentary-moment"
    REVELATION_MOMENT = "revelation-moment"


ENV = os.getenv("ENV", "DEV").upper()
CHANNEL = Channel[os.getenv("CHANNEL", "VIRAL_MOMENT").upper()]

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
    Channel.REVELATION_MOMENT: {
        "queue": BASE_DIR / "viral_moment_queue.csv",
        "results": BASE_DIR / "results" / "revelation-moment",
        "clips": BASE_DIR / "clips" / "revelation-moment",
    },
}

if ENV not in {"DEV", "PROD"}:
    raise ValueError("ENV must be DEV or PROD")

VIDEO_DIR = BASE_DIR / "videos"
QUEUE_FILE = CHANNEL_CONFIG[CHANNEL]["queue"]
RESULTS_DIR = CHANNEL_CONFIG[CHANNEL]["results"]
OUTPUT_DIR = CHANNEL_CONFIG[CHANNEL]["clips"]

STICKER_DIR = BASE_DIR / "sticker"
POP_SFX = BASE_DIR / "sfx" / "pop_click.mp3"

WIDTH, HEIGHT, FPS = 1080, 1920, 30

MIN_DURATION = 30
MAX_DURATION = 90
MAX_VIDEOS_PER_RUN = 1

REVELATION_MIN_DURATION = 30
REVELATION_MAX_DURATION = 60

SCOPES = ["https://www.googleapis.com/auth/drive"]

GDRIVE_SOURCE_FOLDER_ID = os.getenv("GDRIVE_SOURCE_FOLDER_ID")
GDRIVE_JSON_FOLDER_ID = os.getenv("GDRIVE_JSON_FOLDER_ID")
GDRIVE_CLIPS_FOLDER_ID = os.getenv("GDRIVE_CLIPS_FOLDER_ID")

CAPTION_STYLES = [
    caption.CONFIG_STYLE_1,
    caption.CONFIG_STYLE_2,
    caption.CONFIG_STYLE_PUNCHY,
    caption.CONFIG_SAMPLE_STYLE,
    caption.CONFIG_STYLE_STACKED_GRADIENT,
]

ENABLE_CAPTIONS = True
ENABLE_STICKERS = True
ENABLE_POP_SOUND = True

MIN_STICKERS = 1
MAX_STICKERS = 3
STICKER_TARGET_WIDTH = 650
STICKER_MIN_DURATION = 1.5
STICKER_MAX_DURATION = 2.5

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"
}

IMAGE_EXTENSIONS = {
    ".png", ".webp", ".jpg", ".jpeg"
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# INITIALIZATION
# ============================================================

def initialize():
    for folder in [VIDEO_DIR, RESULTS_DIR, OUTPUT_DIR, STICKER_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("UNIVERSAL CLIP GENERATOR")
    print("=" * 60)
    print(f"Environment: {ENV}")
    print(f"Channel:     {CHANNEL.name}")
    print(f"Queue:       {QUEUE_FILE}")
    print(f"Videos:      {VIDEO_DIR}")
    print(f"Results:     {RESULTS_DIR}")
    print(f"Output:      {OUTPUT_DIR}")
    print("=" * 60)


# ============================================================
# GOOGLE DRIVE
# ============================================================

def get_drive():
    credentials_json = os.getenv("GDRIVE_CREDENTIALS", "").strip()
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()

    if credentials_json:
        credentials = Credentials.from_service_account_info(
            json.loads(credentials_json),
            scopes=SCOPES
        )
    elif credentials_path:
        credentials = Credentials.from_service_account_file(
            credentials_path,
            scopes=SCOPES
        )
    else:
        raise RuntimeError("Google Drive credentials are not configured.")

    return build(
        "drive",
        "v3",
        credentials=credentials,
        cache_discovery=False
    )


def find_drive_file(service, folder_id, filename):
    escaped_name = filename.replace("\\", "\\\\").replace("'", "\\'")

    query = (
        f"'{folder_id}' in parents and "
        f"name = '{escaped_name}' and trashed = false"
    )

    response = service.files().list(
        q=query,
        fields="files(id,name,size)",
        pageSize=10,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()

    files = response.get("files", [])
    return files[0] if files else None


def download_drive_file(service, file_id, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temp_path = destination.with_name(destination.name + ".part")

    try:
        request = service.files().get_media(
            fileId=file_id,
            supportsAllDrives=True
        )

        with temp_path.open("wb") as file:
            downloader = MediaIoBaseDownload(file, request)
            done = False

            while not done:
                _, done = downloader.next_chunk()

        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise RuntimeError("Downloaded file is empty.")

        temp_path.replace(destination)
        return destination

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def upload_clip_to_drive(service, clip_path):
    if not GDRIVE_CLIPS_FOLDER_ID:
        raise RuntimeError("GDRIVE_CLIPS_FOLDER_ID is not configured.")

    clip_path = Path(clip_path)

    if not clip_path.exists() or clip_path.stat().st_size == 0:
        raise RuntimeError(f"Clip missing or empty: {clip_path.name}")

    # Avoid duplicate uploads when retrying a partially completed run.
    existing = find_drive_file(
        service,
        GDRIVE_CLIPS_FOLDER_ID,
        clip_path.name
    )

    if existing:
        print(f"[DRIVE] Already uploaded: {clip_path.name}")
        return True

    print(f"[DRIVE] Uploading: {clip_path.name}")

    media = MediaFileUpload(
        str(clip_path),
        mimetype="video/mp4",
        resumable=True
    )

    uploaded = service.files().create(
        body={
            "name": clip_path.name,
            "parents": [GDRIVE_CLIPS_FOLDER_ID]
        },
        media_body=media,
        fields="id,name,size",
        supportsAllDrives=True
    ).execute()

    if not uploaded.get("id"):
        raise RuntimeError(f"Drive upload failed: {clip_path.name}")

    print(f"[DRIVE] Uploaded successfully: {uploaded['name']}")
    return True


def get_result_json(filename, service=None):
    local_path = RESULTS_DIR / filename

    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    if ENV != "PROD":
        return None

    if not service or not GDRIVE_JSON_FOLDER_ID:
        raise RuntimeError("Drive JSON folder or service is not configured.")

    drive_filename = f"{CHANNEL.value}__{filename}"

    drive_file = find_drive_file(
        service,
        GDRIVE_JSON_FOLDER_ID,
        drive_filename
    )

    if not drive_file:
        print(f"[DRIVE] JSON not found: {drive_filename}")
        return None

    print(f"[DRIVE] Downloading JSON: {drive_filename}")

    return download_drive_file(
        service,
        drive_file["id"],
        local_path
    )


# ============================================================
# SOURCE VIDEO DISCOVERY
# ============================================================

def find_source_video(filename):
    requested = Path(filename)
    exact_path = VIDEO_DIR / requested.name

    if exact_path.exists():
        return exact_path

    if not VIDEO_DIR.exists():
        return None

    for path in VIDEO_DIR.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        if path.stem.lower() == requested.stem.lower():
            return path

    return None


def get_source_video(filename, service=None):
    local_path = find_source_video(filename)

    if local_path:
        return local_path

    if ENV != "PROD":
        return None

    if not service or not GDRIVE_SOURCE_FOLDER_ID:
        raise RuntimeError("Drive source folder or service is not configured.")

    drive_filename = Path(filename).name

    drive_file = find_drive_file(
        service,
        GDRIVE_SOURCE_FOLDER_ID,
        drive_filename
    )

    if not drive_file:
        print(f"[DRIVE] Source video not found: {drive_filename}")
        return None

    destination = VIDEO_DIR / drive_filename

    print(f"[DRIVE] Downloading source video: {drive_filename}")

    return download_drive_file(
        service,
        drive_file["id"],
        destination
    )


# ============================================================
# JSON
# ============================================================

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as error:
        print(f"[ERROR] Could not read JSON: {error}")
        return None


# ============================================================
# MOMENT VALIDATION
# ============================================================

def validate_moment(moment, video_duration):
    if CHANNEL == Channel.REVELATION_MOMENT:
        clips = moment.get("clips")

        if not isinstance(clips, list) or not clips:
            print("[SKIP] Revelation moment has no clips.")
            return None

        segments = []
        total_duration = 0

        for index, clip in enumerate(clips, start=1):
            if not isinstance(clip, dict):
                print(f"[SKIP] Invalid segment object: {index}")
                return None

            try:
                start = float(clip["start"])
                end = float(clip["end"])
            except (KeyError, TypeError, ValueError):
                print(f"[SKIP] Invalid timestamps in segment {index}.")
                return None

            if not math.isfinite(start) or not math.isfinite(end):
                print(f"[SKIP] Non-finite timestamps in segment {index}.")
                return None

            if start < 0 or end <= start or start >= video_duration:
                print(
                    f"[SKIP] Invalid segment {index}: "
                    f"{start:.2f}s - {end:.2f}s"
                )
                return None

            end = min(end, video_duration)
            segment_duration = end - start

            if segment_duration <= 0:
                return None

            segments.append((start, end))
            total_duration += segment_duration

        if not (
            REVELATION_MIN_DURATION
            <= total_duration
            <= REVELATION_MAX_DURATION
        ):
            print(
                f"[SKIP] Revelation duration: {total_duration:.2f}s "
                f"(allowed: {REVELATION_MIN_DURATION}-"
                f"{REVELATION_MAX_DURATION}s)"
            )
            return None

        print(
            f"[VALIDATION] Revelation segments: {len(segments)} | "
            f"Total duration: {total_duration:.2f}s"
        )

        return segments

    try:
        start = float(moment["start"])
        end = float(moment["end"])
    except (KeyError, TypeError, ValueError):
        print("[SKIP] Moment has invalid start/end timestamps.")
        return None

    if not math.isfinite(start) or not math.isfinite(end):
        print("[SKIP] Moment has non-finite timestamps.")
        return None

    start = max(0.0, start)
    end = min(end, video_duration)
    duration = end - start

    if start >= video_duration:
        print("[SKIP] Moment starts beyond source video.")
        return None

    if duration < MIN_DURATION or duration > MAX_DURATION:
        print(f"[SKIP] Invalid duration: {duration:.2f}s")
        return None

    return [(start, end)]


# ============================================================
# VERTICAL VIDEO
# ============================================================

def make_vertical_video(clip):
    scale = max(WIDTH / clip.w, HEIGHT / clip.h)
    resized = clip.resized(scale)

    return resized.cropped(
        x_center=resized.w / 2,
        y_center=resized.h / 2,
        width=WIDTH,
        height=HEIGHT
    )


# ============================================================
# CAPTIONS
# ============================================================

def create_captions(audio_path, duration, highlight_words):
    if not ENABLE_CAPTIONS:
        return None

    config = random.choice(CAPTION_STYLES).copy()

    if hasattr(caption, "SPECIAL_WORDS"):
        caption.SPECIAL_WORDS.update(
            str(word).strip().upper()
            for word in highlight_words
            if str(word).strip()
        )

    print(f"[CAPTIONS] Style: {config['style']}")
    print(f"[CAPTIONS] AI highlights: {highlight_words}")

    try:
        caption_clip = caption.generate_caption_overlay(
            audio_path=str(audio_path),
            config=config
        )

        if caption_clip is not None:
            caption_clip = (
                caption_clip
                .with_start(0)
                .with_duration(duration)
            )
            print("[CAPTIONS] Generated successfully.")

        return caption_clip

    except Exception as error:
        print(f"[CAPTIONS] Generation failed: {error}")
        traceback.print_exc()
        return None


# ============================================================
# STICKERS
# ============================================================

def get_sticker_files():
    if not STICKER_DIR.exists():
        return []

    return [
        path for path in STICKER_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def popup_scale(t, duration):
    pop_duration = min(0.30, duration)

    if t <= 0:
        return 0.05

    if t < pop_duration:
        progress = t / pop_duration
        return 1.15 * progress + 0.05 * progress * progress

    return 1.0 + 0.015 * np.sin(t * 2.0)


def create_one_sticker(path, video_duration):
    base = None

    try:
        with Image.open(path) as image:
            image = image.convert("RGBA")
            original_w, original_h = image.size

            if original_w <= 0 or original_h <= 0:
                raise ValueError("Invalid sticker dimensions.")

            scale = STICKER_TARGET_WIDTH / original_w
            new_w = STICKER_TARGET_WIDTH
            new_h = max(1, round(original_h * scale))

            image = image.resize(
                (new_w, new_h),
                Image.Resampling.LANCZOS
            )

            image_array = np.array(image)

        base = ImageClip(image_array, transparent=True)
        original_w, original_h = base.size

        max_start = max(0, video_duration - STICKER_MIN_DURATION)
        start = random.uniform(0, max_start)

        duration = min(
            random.uniform(STICKER_MIN_DURATION, STICKER_MAX_DURATION),
            video_duration - start
        )

        if duration <= 0:
            base.close()
            return None, None

        side = random.choice(["left", "right"])
        vertical = random.choice(["top", "middle"])

        x = 65 if side == "left" else WIDTH - original_w - 65
        y = 260 if vertical == "top" else 720

        x = max(15, min(x, WIDTH - original_w - 15))
        y = max(100, min(y, HEIGHT - original_h - 100))

        def scale_at(t):
            return max(0.05, popup_scale(t, duration))

        def position_at(t):
            scale = scale_at(t)
            current_w = original_w * scale
            current_h = original_h * scale

            return (
                x + (original_w - current_w) / 2,
                y + (original_h - current_h) / 2
            )

        sticker = (
            base
            .resized(scale_at)
            .with_start(start)
            .with_duration(duration)
            .with_position(position_at)
            .with_effects([FadeIn(min(0.15, duration / 2))])
        )

        print(
            f"[STICKER] {path.name} | "
            f"Start: {start:.2f}s | Duration: {duration:.2f}s"
        )

        return sticker, start

    except Exception as error:
        print(f"[STICKER] Failed: {path.name}: {error}")
        traceback.print_exc()

        if base is not None:
            try:
                base.close()
            except Exception:
                pass

        return None, None


def create_stickers(duration):
    if not ENABLE_STICKERS:
        return [], []

    files = get_sticker_files()

    if not files:
        print("[STICKERS] No sticker images found.")
        return [], []

    count = min(
        random.randint(MIN_STICKERS, MAX_STICKERS),
        max(0, int(duration // 3))
    )

    selected = random.sample(files, min(count, len(files)))
    sticker_clips = []
    sticker_times = []

    print(f"[STICKERS] Creating {len(selected)} popup stickers.")

    for path in selected:
        sticker, start = create_one_sticker(path, duration)

        if sticker is not None:
            sticker_clips.append(sticker)
            sticker_times.append(start)

    return sticker_clips, sticker_times


# ============================================================
# AUDIO
# ============================================================

def extract_audio(clip, output_path):
    if clip.audio is None:
        print("[WARNING] Source video has no audio.")
        return False

    try:
        clip.audio.write_audiofile(
            str(output_path),
            fps=16000,
            nbytes=2,
            codec="pcm_s16le",
            logger=None
        )
        return True

    except Exception as error:
        print(f"[AUDIO] Extraction failed: {error}")
        return False


def create_final_audio(source_audio, sticker_times, sfx_clips):
    audio_layers = []

    if source_audio is not None:
        audio_layers.append(source_audio)

    if ENABLE_POP_SOUND and POP_SFX.exists():
        for start in sticker_times:
            try:
                sfx = AudioFileClip(str(POP_SFX)).with_start(start)
                sfx_clips.append(sfx)
                audio_layers.append(sfx)
            except Exception as error:
                print(f"[SFX] Could not load pop sound: {error}")

    if not audio_layers:
        return None

    return CompositeAudioClip(audio_layers)


# ============================================================
# GENERATE ONE MOMENT
# ============================================================

def generate_clip(source_path, moment, video_id, moment_index, global_highlights):
    source = None
    moment_clip = None
    vertical = None
    captions = None
    final = None
    final_audio = None

    segment_clips = []
    sticker_clips = []
    sfx_clips = []

    title = moment.get("title", "Untitled")
    moment_highlights = moment.get("highlight_words", [])

    if not isinstance(moment_highlights, list):
        moment_highlights = []

    highlight_words = list(dict.fromkeys(
        [*global_highlights, *moment_highlights]
    ))

    print("\n" + "=" * 60)
    print(f"PROCESSING MOMENT {moment_index}")
    print(f"TITLE: {title}")
    print(f"HOOK: {moment.get('hook', 'N/A')}")
    print(f"HIGHLIGHTS: {highlight_words}")
    print("=" * 60)

    output_name = f"{Path(video_id).stem}_{moment_index:02d}"
    output_path = OUTPUT_DIR / f"{output_name}.mp4"
    temp_output_path = OUTPUT_DIR / f"{output_name}.temp.mp4"

    # An existing valid output counts as successfully generated.
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[SKIP] Already generated: {output_path.name}")
        return True

    try:
        source = VideoFileClip(str(source_path))
        segments = validate_moment(moment, source.duration)

        if not segments:
            return False

        for index, (start, end) in enumerate(segments, start=1):
            print(
                f"[SEGMENT {index}] "
                f"{start:.2f}s - {end:.2f}s "
                f"({end - start:.2f}s)"
            )

            segment_clips.append(source.subclipped(start, end))

        if len(segment_clips) == 1:
            moment_clip = segment_clips[0]
        else:
            moment_clip = concatenate_videoclips(
                segment_clips,
                method="compose"
            )

        duration = moment_clip.duration

        print(f"[VIDEO] Segments: {len(segment_clips)}")
        print(f"[VIDEO] Combined duration: {duration:.2f}s")

        vertical = make_vertical_video(moment_clip)

        with tempfile.TemporaryDirectory(prefix="viral_clip_") as temp:
            audio_path = Path(temp) / "audio.wav"
            overlays = []

            if extract_audio(moment_clip, audio_path):
                captions = create_captions(
                    audio_path,
                    duration,
                    highlight_words
                )

                if captions is not None:
                    overlays.append(captions)

            sticker_clips, sticker_times = create_stickers(duration)
            overlays.extend(sticker_clips)

            final = CompositeVideoClip(
                [vertical, *overlays],
                size=(WIDTH, HEIGHT)
            ).with_duration(duration)

            final_audio = create_final_audio(
                moment_clip.audio,
                sticker_times,
                sfx_clips
            )

            if final_audio is not None:
                final = final.with_audio(final_audio)

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            print("[EXPORT] Rendering vertical video...")

            final.write_videofile(
                str(temp_output_path),
                fps=FPS,
                codec="libx264",
                audio_codec="aac",
                preset="medium",
                threads=4,
                logger="bar"
            )

        if (
            not temp_output_path.exists()
            or temp_output_path.stat().st_size == 0
        ):
            raise RuntimeError("Video export failed or produced an empty file.")

        temp_output_path.replace(output_path)

        print(f"[SUCCESS] Created: {output_path}")
        return True

    except Exception as error:
        print(f"[ERROR] Clip generation failed: {error}")
        traceback.print_exc()

        if temp_output_path.exists():
            try:
                temp_output_path.unlink()
            except Exception:
                pass

        return False

    finally:
        clips_to_close = [
            final,
            final_audio,
            captions,
            *sticker_clips,
            *sfx_clips,
            vertical,
            moment_clip,
            *segment_clips,
            source
        ]

        seen = set()

        for clip in clips_to_close:
            if clip is not None and id(clip) not in seen:
                seen.add(id(clip))

                try:
                    clip.close()
                except Exception:
                    pass


# ============================================================
# PROCESS ONE JSON FILE
# ============================================================

def process_result(json_path, video_override=None, service=None):
    print(f"\n[JSON] Processing: {json_path.name}")

    data = load_json(json_path)

    if not isinstance(data, dict):
        print("[ERROR] Invalid JSON.")
        return False, []

    video_filename = video_override or data.get("video")

    if not isinstance(video_filename, str) or not video_filename.strip():
        print("[ERROR] Video filename missing from JSON.")
        return False, []

    source_path = get_source_video(video_filename, service)

    if source_path is None:
        print(f"[ERROR] Source video not found: {video_filename}")
        return False, []

    print(f"[VIDEO] Found: {source_path.name}")

    moments = data.get("moments", [])

    if not isinstance(moments, list) or not moments:
        print("[ERROR] No moments found in JSON.")
        return False, []

    global_highlights = data.get("highlight_words", [])

    if not isinstance(global_highlights, list):
        global_highlights = []

    all_successful = True
    expected_files = []

    for index, moment in enumerate(moments, start=1):
        output_path = OUTPUT_DIR / (
            f"{Path(video_filename).stem}_{index:02d}.mp4"
        )
        expected_files.append(output_path)

        if not isinstance(moment, dict):
            print(f"[ERROR] Moment {index} is not a valid object.")
            all_successful = False
            continue

        success = generate_clip(
            source_path=source_path,
            moment=moment,
            video_id=video_filename,
            moment_index=index,
            global_highlights=global_highlights
        )

        if not success:
            all_successful = False

    # Verify every expected MP4 exists and is non-empty.
    for path in expected_files:
        if not path.exists() or path.stat().st_size == 0:
            print(f"[ERROR] Expected clip missing: {path.name}")
            all_successful = False

    generated = sum(
        1 for path in expected_files
        if path.exists() and path.stat().st_size > 0
    )

    print(
        f"\n[SUMMARY] {json_path.name} | "
        f"Expected: {len(expected_files)} | "
        f"Available: {generated} | "
        f"Success: {all_successful}"
    )

    return all_successful, expected_files


# ============================================================
# CSV QUEUE
# ============================================================

def read_queue():
    if not QUEUE_FILE.exists():
        raise FileNotFoundError(f"Queue file not found: {QUEUE_FILE}")

    with QUEUE_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    required_columns = {
        "video_title",
        "json_done",
        "post_done",
        "clip_done"
    }

    missing = required_columns - set(fieldnames)

    if missing:
        raise RuntimeError(
            f"CSV is missing columns: {', '.join(sorted(missing))}"
        )

    return fieldnames, rows


def update_clip_done(video_filename):
    fieldnames, rows = read_queue()
    updated = False

    for row in rows:
        if (row.get("video_title") or "").strip() == video_filename:
            row["clip_done"] = "true"
            updated = True
            break

    if not updated:
        raise RuntimeError(
            f"Could not find video in CSV to update: {video_filename}"
        )

    # Write atomically to avoid leaving a partially written CSV.
    temp_path = QUEUE_FILE.with_suffix(".tmp")

    with temp_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)

    temp_path.replace(QUEUE_FILE)

    print(f"[QUEUE] clip_done=true saved for: {video_filename}")


def select_next_video(rows, service):
    for row in rows:
        video = (row.get("video_title") or "").strip()

        if not video:
            continue

        if (row.get("json_done") or "").strip().lower() != "true":
            continue

        if (row.get("clip_done") or "").strip().lower() == "true":
            continue

        json_filename = f"{Path(video).stem}.json"
        json_path = RESULTS_DIR / json_filename

        if not json_path.exists():
            json_path = get_result_json(json_filename, service)

        if not json_path or not json_path.exists():
            print(f"[SKIP] JSON missing: {json_filename}")
            continue

        return video, json_path

    return None, None


# ============================================================
# UPLOAD AND VERIFY ALL CLIPS
# ============================================================

def upload_all_clips(service, clip_paths):
    if not clip_paths:
        raise RuntimeError("No clips were generated.")

    if not GDRIVE_CLIPS_FOLDER_ID:
        raise RuntimeError("GDRIVE_CLIPS_FOLDER_ID is not configured.")

    for clip_path in clip_paths:
        if not clip_path.exists() or clip_path.stat().st_size == 0:
            raise RuntimeError(f"Clip missing or empty: {clip_path.name}")

        upload_clip_to_drive(service, clip_path)

    # Verify every expected file exists in the destination Drive folder.
    for clip_path in clip_paths:
        drive_file = find_drive_file(
            service,
            GDRIVE_CLIPS_FOLDER_ID,
            clip_path.name
        )

        if not drive_file:
            raise RuntimeError(
                f"Drive verification failed: {clip_path.name}"
            )

        print(f"[DRIVE] Verified: {clip_path.name}")

    print(f"[DRIVE] All {len(clip_paths)} clips verified.")
    return True


# ============================================================
# MAIN
# ============================================================

def resolve_json_path(filename):
    path = Path(filename)

    if path.is_absolute():
        return path

    if path.exists():
        return path

    return RESULTS_DIR / path.name


def main(json_file=None, video_file=None):
    initialize()

    if not hasattr(caption, "generate_caption_overlay"):
        raise RuntimeError(
            "caption_generator.py must contain generate_caption_overlay()."
        )

    service = get_drive() if ENV == "PROD" else None

    # Manual single-JSON processing.
    # Manual mode does not update the queue's clip_done column.
    if json_file:
        json_path = resolve_json_path(json_file)

        if not json_path.exists() and ENV == "PROD":
            json_path = get_result_json(
                Path(json_file).name,
                service
            )

        if not json_path or not json_path.exists():
            raise RuntimeError(f"JSON file not found: {json_file}")

        success, expected_files = process_result(
            json_path,
            video_override=video_file,
            service=service
        )

        if not success:
            raise RuntimeError("One or more clips failed to generate.")

        if ENV == "PROD":
            upload_all_clips(service, expected_files)

        return

    # Automatic queue processing: ONE source video per run.
    fieldnames, rows = read_queue()

    video, json_path = select_next_video(rows, service)

    if not video:
        print("[QUEUE] No eligible source videos found.")
        return

    print("\n" + "=" * 60)
    print(f"[QUEUE] SELECTED VIDEO: {video}")
    print("[QUEUE] Processing only this source video.")
    print("=" * 60)

    success, expected_files = process_result(
        json_path,
        video_override=video,
        service=service
    )

    if not success:
        raise RuntimeError(
            f"Clip generation incomplete for {video}. "
            "clip_done remains false."
        )

    # In production, upload and verify every expected MP4 first.
    if ENV == "PROD":
        upload_all_clips(service, expected_files)
    else:
        print(
            "[DEV] Clips generated locally. "
            "Drive upload and clip_done update are skipped."
        )
        return

    # Update CSV only after successful generation AND Drive verification.
    update_clip_done(video)

    print("\n" + "=" * 60)
    print(f"[SUCCESS] Completed: {video}")
    print("[SUCCESS] All clips generated and verified in Google Drive.")
    print("[SUCCESS] clip_done=true")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json",
        help="Process a specific JSON file"
    )

    parser.add_argument(
        "--video",
        help="Source video filename"
    )

    args = parser.parse_args()

    main(args.json, args.video)
