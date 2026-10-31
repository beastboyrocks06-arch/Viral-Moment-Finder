
import argparse
import json
import random
import tempfile
import traceback
from pathlib import Path
import csv
import numpy as np
from PIL import Image

from moviepy import (
    VideoFileClip,
    ImageClip,
    CompositeVideoClip,
    CompositeAudioClip,
)
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.fx import FadeIn

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Add project root to Python import path
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import caption_generator as caption

VIDEO_DIR = BASE_DIR / "videos"
RESULTS_DIR = BASE_DIR / "viral_moments" / "results"
OUTPUT_DIR = BASE_DIR / "clips"
STICKER_DIR = BASE_DIR / "sticker"
POP_SFX = BASE_DIR / "sfx" / "pop_click.mp3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
QUEUE_FILE = BASE_DIR / "viral_video_queue.csv"
WIDTH, HEIGHT, FPS = 1080, 1920, 30
MIN_DURATION, MAX_DURATION = 20, 60

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
STICKER_MAX_WIDTH = 700
STICKER_TARGET_WIDTH = 650
STICKER_MIN_DURATION = 1.5
STICKER_MAX_DURATION = 2.5

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"
}
IMAGE_EXTENSIONS = {".png", ".webp", ".jpg", ".jpeg"}


# ============================================================
# INITIALIZATION
# ============================================================

def initialize():
    for folder in [VIDEO_DIR, RESULTS_DIR, OUTPUT_DIR, STICKER_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("VIRAL MOMENT CLIP GENERATOR")
    print("=" * 60)
    print(f"Videos:  {VIDEO_DIR}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Stickers: {STICKER_DIR}")


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_source_video(filename):
    requested = Path(filename)

    # Exact filename match
    exact_path = VIDEO_DIR / requested.name
    if exact_path.exists():
        return exact_path

    # Match by filename without extension
    for path in VIDEO_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            if path.stem.lower() == requested.stem.lower():
                return path

    return None


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
    try:
        start = float(moment["start"])
        end = float(moment["end"])
    except (KeyError, TypeError, ValueError):
        print("[SKIP] Moment has invalid start/end timestamps.")
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

    return start, end


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
# CAPTION GENERATION
# ============================================================

def create_captions(audio_path, duration, highlight_words):
    if not ENABLE_CAPTIONS:
        return None

    config = random.choice(CAPTION_STYLES).copy()

    # Add AI-selected words to the existing special-word system.
    # These words will receive the special font/color treatment.
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
# STICKER ANIMATION
# ============================================================

def get_sticker_files():
    if not STICKER_DIR.exists():
        return []

    return [
        path for path in STICKER_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def popup_scale(t, duration):
    """Large, attention-grabbing popup animation."""

    pop_duration = min(0.30, duration)

    if t <= 0:
        return 0.05

    if t < pop_duration:
        progress = t / pop_duration

        # Pop to 115%, then settle to 100%
        return (
            1.15 * progress
            + 0.05 * progress * progress
        )

    # Subtle breathing
    return 1.0 + 0.015 * np.sin(t * 2.0)

def create_one_sticker(path, video_duration):
    base = None

    try:
        # -----------------------------------------------------
        # 1. LOAD IMAGE
        # -----------------------------------------------------

        with Image.open(path) as image:
            image = image.convert("RGBA")

            original_w, original_h = image.size

            if original_w <= 0 or original_h <= 0:
                raise ValueError("Invalid sticker dimensions")

            # -------------------------------------------------
            # 2. RESIZE TO TARGET DISPLAY WIDTH
            # -------------------------------------------------

            scale = STICKER_TARGET_WIDTH / original_w

            new_w = STICKER_TARGET_WIDTH
            new_h = max(1, round(original_h * scale))

            image = image.resize(
                (new_w, new_h),
                Image.Resampling.LANCZOS
            )

            image_array = np.array(image)

        # -----------------------------------------------------
        # 3. CREATE MOVIEPY CLIP
        # -----------------------------------------------------

        base = ImageClip(
            image_array,
            transparent=True
        )

        original_w, original_h = base.size

        # -----------------------------------------------------
        # 4. RANDOM TIMING
        # -----------------------------------------------------

        max_start = max(
            0,
            video_duration - STICKER_MIN_DURATION
        )

        start = random.uniform(0, max_start)

        duration = min(
            random.uniform(
                STICKER_MIN_DURATION,
                STICKER_MAX_DURATION
            ),
            video_duration - start
        )

        if duration <= 0:
            base.close()
            return None, None

        # -----------------------------------------------------
        # 5. POSITION
        # -----------------------------------------------------

        side = random.choice(["left", "right"])
        vertical = random.choice(["top", "middle"])

        x = (
            65
            if side == "left"
            else WIDTH - original_w - 65
        )

        y = (
            260
            if vertical == "top"
            else 720
        )

        # Keep sticker inside frame
        x = max(
            15,
            min(x, WIDTH - original_w - 15)
        )

        y = max(
            100,
            min(y, HEIGHT - original_h - 100)
        )

        # -----------------------------------------------------
        # 6. POPUP ANIMATION
        # -----------------------------------------------------

        def scale_at(t):
            return max(
                0.05,
                popup_scale(t, duration)
            )

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
            .with_effects([
                FadeIn(min(0.15, duration / 2))
            ])
        )

        print(
            f"[STICKER] Loaded: {path.name} | "
            f"Original: {original_w}x{original_h} | "
            f"Target width: {STICKER_TARGET_WIDTH}px | "
            f"Start: {start:.2f}s"
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
    sticker_clips = []
    sfx_clips = []
    final_audio = None

    title = moment.get("title", "Untitled")
    moment_highlights = moment.get("highlight_words", [])
    highlight_words = list(dict.fromkeys(
        [*global_highlights, *moment_highlights]
    ))

    print("\n" + "=" * 60)
    print(f"PROCESSING MOMENT {moment_index}")
    print(f"TITLE: {title}")
    print(f"HOOK: {moment.get('hook', 'N/A')}")
    print(f"HIGHLIGHTS: {highlight_words}")
    print("=" * 60)

    output_path = OUTPUT_DIR / f"{Path(video_id).stem}_{moment_index:02d}.mp4"

    if output_path.exists():
        print(f"[SKIP] Already exists: {output_path.name}")
        return False

    try:
        source = VideoFileClip(str(source_path))

        timestamps = validate_moment(moment, source.duration)
        if not timestamps:
            return False

        start, end = timestamps
        moment_clip = source.subclipped(start, end)
        duration = moment_clip.duration

        print(f"[VIDEO] Source: {start:.2f}s - {end:.2f}s")
        print(f"[VIDEO] Clip duration: {duration:.2f}s")

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

            output_path.parent.mkdir(parents=True, exist_ok=True)

            print("[EXPORT] Rendering vertical video...")

            final.write_videofile(
                str(output_path),
                fps=FPS,
                codec="libx264",
                audio_codec="aac",
                preset="medium",
                threads=4,
                logger="bar"
            )

        print(f"[SUCCESS] Created: {output_path}")
        return True

    except Exception as error:
        print(f"[ERROR] Clip generation failed: {error}")
        traceback.print_exc()

        if output_path.exists():
            output_path.unlink()

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

def process_result(json_path, video_override=None):
    print(f"\n[JSON] Processing: {json_path.name}")

    data = load_json(json_path)
    if not data:
        return

    video_filename = video_override or data.get("video")

    if not isinstance(video_filename, str) or not video_filename.strip():
        print("[ERROR] Video filename missing from JSON.")
        return

    source_path = find_source_video(video_filename)

    if source_path is None:
        print(f"[ERROR] Source video not found: {video_filename}")
        return

    print(f"[VIDEO] Found: {source_path.name}")

    moments = data.get("moments", [])

    if not isinstance(moments, list) or not moments:
        print("[INFO] No moments found in JSON.")
        return

    global_highlights = data.get("highlight_words", [])
    if not isinstance(global_highlights, list):
        global_highlights = []

    generated = skipped = 0

    for index, moment in enumerate(moments, start=1):
        if not isinstance(moment, dict):
            skipped += 1
            continue

        success = generate_clip(
            source_path=source_path,
            moment=moment,
            video_id=video_filename,
            moment_index=index,
            global_highlights=global_highlights
        )

        if success:
            generated += 1
        else:
            skipped += 1

    print(
        f"\n[SUMMARY] {json_path.name} | "
        f"Generated: {generated} | Skipped: {skipped}"
    )


# ============================================================
# MAIN
# ============================================================
def main(json_file=None, video_file=None):
    initialize()

    if not hasattr(caption, "generate_caption_overlay"):
        raise RuntimeError(
            "caption_generator.py must contain generate_caption_overlay()."
        )

    # Manual single-JSON processing
    if json_file:
        json_path = Path(json_file)
        if not json_path.is_absolute():
            json_path = RESULTS_DIR / json_path

        if not json_path.exists():
            print(f"[ERROR] JSON file not found: {json_path}")
            return

        process_result(json_path, video_file)
        return

    # Automatic CSV queue processing
    with QUEUE_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        video = row["video_title"].strip()

        if row["json_done"].strip().lower() != "true":
            continue

        if row["post_done"].strip().lower() == "true":
            continue

        json_path = RESULTS_DIR / f"{Path(video).stem}.json"

        if not json_path.exists():
            print(f"[SKIP] JSON missing: {json_path.name}")
            continue

        print(f"\n[QUEUE] Processing: {video}")

        try:
            process_result(json_path, video)
        except KeyboardInterrupt:
            print("\n[STOP] Interrupted.")
            break
        except Exception as error:
            print(f"[ERROR] {video}: {error}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("CLIP GENERATION QUEUE FINISHED")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="Process a specific JSON file")
    parser.add_argument("--video", help="Source video filename")
    args = parser.parse_args()
    main(args.json, args.video)