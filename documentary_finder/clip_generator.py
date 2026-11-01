
import os
import sys
import json
import random
import re
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from moviepy import (
    VideoFileClip,
    ImageClip,
    CompositeVideoClip,
    CompositeAudioClip,
    concatenate_videoclips,
)
from moviepy.audio.io.AudioFileClip import AudioFileClip


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_DIR = BASE_DIR / "videos"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = BASE_DIR / "clips"
STICKER_DIR = BASE_DIR / "sticker"
SFX_DIR = BASE_DIR / "sfx"

POP_SFX = SFX_DIR / "pop_click.mp3"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# VIDEO SETTINGS
# ============================================================

WIDTH = 1080
HEIGHT = 1920
FPS = 30

MIN_CLIP_DURATION = 20
MAX_CLIP_DURATION = 60

VIDEO_CODEC = "libx264"
AUDIO_CODEC = "aac"
EXPORT_PRESET = "medium"

# Center crop source video to vertical 9:16.
# This fills the frame without black bars.
VERTICAL_CROP = True


# ============================================================
# CAPTION SETTINGS
# ============================================================

ENABLE_CAPTIONS = True

# Automatically searches this project and parent directories.
CAPTION_MODULE = None


# ============================================================
# STICKER SETTINGS
# ============================================================

ENABLE_STICKERS = True

STICKER_TARGET_WIDTH = 350
STICKER_MIN_DURATION = 1.5
STICKER_MAX_DURATION = 2.5

MAX_STICKERS_PER_MOMENT = 2


# ============================================================
# SOUND EFFECT SETTINGS
# ============================================================

ENABLE_POP_SOUND = True
MAX_POP_SOUNDS = 2
POP_VOLUME = 0.45


# ============================================================
# LOAD EXISTING CAPTION GENERATOR
# ============================================================

def load_caption_generator():

    possible_paths = [
        BASE_DIR / "caption_generator.py",
        BASE_DIR.parent / "caption_generator.py",
        BASE_DIR.parent.parent / "caption_generator.py",
    ]

    for path in possible_paths:

        if not path.exists():
            continue

        try:
            import importlib.util

            if str(path.parent) not in sys.path:
                sys.path.insert(0, str(path.parent))

            spec = importlib.util.spec_from_file_location(
                "coldcases_caption_generator",
                path
            )

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "generate_caption_overlay"):
                print(f"[CAPTION] Loaded: {path}")
                return module

        except Exception as error:
            print(f"[CAPTION] Could not load {path}: {error}")

    print("[CAPTION] No compatible caption_generator.py found.")
    return None


if ENABLE_CAPTIONS:
    CAPTION_MODULE = load_caption_generator()


# ============================================================
# HELPERS
# ============================================================

def safe_filename(value):

    value = re.sub(r'[<>:"/\\|?*]', "", str(value))
    value = re.sub(r"\s+", "_", value.strip())

    return value[:100]


def find_video(video_name):

    possible_paths = [
        VIDEO_DIR / video_name,
        BASE_DIR / video_name,
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # Search recursively if the video isn't in the expected folder.
    for path in BASE_DIR.rglob(video_name):
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"Could not find source video: {video_name}\n"
        f"Expected location: {VIDEO_DIR}"
    )


def find_json_file():

    json_files = list(RESULTS_DIR.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found inside {RESULTS_DIR}"
        )

    # Use the most recently modified JSON file.
    return max(json_files, key=lambda path: path.stat().st_mtime)


def make_vertical(clip):

    if not VERTICAL_CROP:
        return clip.resized(new_size=(WIDTH, HEIGHT))

    original_w, original_h = clip.size

    scale = max(
        WIDTH / original_w,
        HEIGHT / original_h
    )

    resized = clip.resized(scale)

    new_w, new_h = resized.size

    x1 = max(0, (new_w - WIDTH) // 2)
    y1 = max(0, (new_h - HEIGHT) // 2)

    return resized.cropped(
        x1=x1,
        y1=y1,
        x2=x1 + WIDTH,
        y2=y1 + HEIGHT
    )


# ============================================================
# EXTRACT SOURCE SEGMENTS
# ============================================================

def extract_moment_clips(source_video, moment):

    clips = []

    moment_id = moment.get("moment_id", "UNKNOWN")

    source_segments = moment.get("clips", [])

    if not source_segments:
        print(f"[WARNING] {moment_id} has no source clips.")
        return clips

    for segment in source_segments:

        clip_id = segment.get("clip_id", "UNKNOWN")

        try:
            start = float(segment["start"])
            end = float(segment["end"])

            if start < 0 or end <= start:
                print(f"[SKIP] Invalid timestamps: {clip_id}")
                continue

            if end > source_video.duration:
                print(
                    f"[SKIP] {clip_id}: End timestamp "
                    f"{end:.2f}s exceeds video duration."
                )
                continue

            duration = end - start

            print(
                f"[CUT] {clip_id} | "
                f"{start:.2f}s - {end:.2f}s | "
                f"{duration:.2f}s | "
                f"Role: {segment.get('role', 'Unknown')}"
            )

            source_clip = source_video.subclipped(start, end)

            vertical_clip = make_vertical(source_clip)

            clips.append(vertical_clip)

        except Exception as error:

            print(f"[ERROR] Could not extract {clip_id}: {error}")
            traceback.print_exc()

    return clips


# ============================================================
# STICKER GENERATOR
# ============================================================

def create_stickers(video_duration):

    if not ENABLE_STICKERS:
        return []

    if not STICKER_DIR.exists():
        return []

    sticker_files = (
        list(STICKER_DIR.glob("*.png"))
        + list(STICKER_DIR.glob("*.jpg"))
        + list(STICKER_DIR.glob("*.webp"))
    )

    if not sticker_files:
        return []

    random.shuffle(sticker_files)

    count = min(
        len(sticker_files),
        MAX_STICKERS_PER_MOMENT
    )

    sticker_clips = []

    for path in sticker_files[:count]:

        try:
            with Image.open(path) as image:

                image = image.convert("RGBA")

                # Remove transparent margins so the actual artwork
                # uses the available display width.
                bbox = image.getchannel("A").getbbox()

                if bbox:
                    image = image.crop(bbox)

                original_w, original_h = image.size

                if original_w <= 0 or original_h <= 0:
                    continue

                scale = STICKER_TARGET_WIDTH / original_w

                new_size = (
                    STICKER_TARGET_WIDTH,
                    max(1, round(original_h * scale))
                )

                image = image.resize(
                    new_size,
                    Image.Resampling.LANCZOS
                )

                image_array = np.array(image)

            sticker = ImageClip(
                image_array,
                transparent=True
            )

            duration = min(
                random.uniform(
                    STICKER_MIN_DURATION,
                    STICKER_MAX_DURATION
                ),
                video_duration
            )

            if duration <= 0:
                sticker.close()
                continue

            start = random.uniform(
                0,
                max(0, video_duration - duration)
            )

            sticker_w, sticker_h = sticker.size

            side = random.choice(["left", "right"])
            vertical = random.choice(["top", "middle"])

            x = (
                45
                if side == "left"
                else WIDTH - sticker_w - 45
            )

            y = 250 if vertical == "top" else 750

            x = max(15, min(x, WIDTH - sticker_w - 15))
            y = max(100, min(y, HEIGHT - sticker_h - 100))

            sticker = (
                sticker
                .with_start(start)
                .with_duration(duration)
                .with_position((x, y))
            )

            sticker_clips.append(sticker)

            print(
                f"[STICKER] {path.name} | "
                f"Size: {sticker_w}x{sticker_h} | "
                f"Start: {start:.2f}s"
            )

        except Exception as error:

            print(f"[STICKER ERROR] {path.name}: {error}")

    return sticker_clips


# ============================================================
# POP SOUND EFFECTS
# ============================================================

def create_pop_sounds(video_duration):

    if not ENABLE_POP_SOUND or not POP_SFX.exists():
        return []

    sound_clips = []

    try:

        count = min(
            MAX_POP_SOUNDS,
            max(1, int(video_duration // 15))
        )

        for _ in range(count):

            start = random.uniform(
                1,
                max(1.1, video_duration - 1)
            )

            sound = AudioFileClip(str(POP_SFX))

            sound = (
                sound
                .with_start(start)
                .with_volume_scaled(POP_VOLUME)
            )

            sound_clips.append(sound)

            print(f"[SFX] Pop sound at {start:.2f}s")

    except Exception as error:

        print(f"[SFX ERROR] {error}")

    return sound_clips


# ============================================================
# CAPTION GENERATION
# ============================================================

def add_captions(final_video, output_name):

    if not ENABLE_CAPTIONS or CAPTION_MODULE is None:
        print("[CAPTION] Skipping captions.")
        return final_video

    if final_video.audio is None:
        print("[CAPTION] Video has no audio. Skipping captions.")
        return final_video

    temp_audio = OUTPUT_DIR / f"{output_name}_temp_audio.wav"

    try:

        print("[CAPTION] Extracting final audio...")

        final_video.audio.write_audiofile(
            str(temp_audio),
            fps=44100,
            nbytes=2,
            codec="pcm_s16le",
            logger=None
        )

        print("[CAPTION] Generating caption overlay...")

        caption_overlay = (
            CAPTION_MODULE.generate_caption_overlay(
                str(temp_audio)
            )
        )

        if caption_overlay is None:
            print("[CAPTION] Generator returned no overlay.")
            return final_video

        caption_overlay = caption_overlay.with_duration(
            final_video.duration
        )

        composed = CompositeVideoClip(
            [final_video, caption_overlay],
            size=(WIDTH, HEIGHT)
        ).with_audio(final_video.audio)

        print("[CAPTION] Captions added.")

        return composed

    except Exception as error:

        print(f"[CAPTION ERROR] {error}")
        traceback.print_exc()

        return final_video

    finally:

        if temp_audio.exists():
            try:
                temp_audio.unlink()
            except Exception:
                pass


# ============================================================
# GENERATE ONE MOMENT
# ============================================================

def generate_moment(source_video, moment):

    moment_id = moment.get("moment_id", "UNKNOWN")
    title = moment.get("title", moment_id)

    output_name = safe_filename(
        f"{moment_id}_{title}"
    )

    output_path = OUTPUT_DIR / f"{output_name}.mp4"

    print("\n" + "=" * 65)
    print(f"[MOMENT] {moment_id}: {title}")
    print("=" * 65)

    source_clips = []
    sticker_clips = []
    sound_clips = []
    final_video = None
    captioned_video = None

    try:

        # 1. CUT SOURCE SEGMENTS
        source_clips = extract_moment_clips(
            source_video,
            moment
        )

        if not source_clips:
            print("[ERROR] No valid source clips.")
            return None

        # 2. JOIN SEGMENTS IN JSON ORDER
        print("[JOIN] Combining source clips...")

        final_video = concatenate_videoclips(
            source_clips,
            method="compose"
        )

        print(
            f"[JOIN] Combined duration: "
            f"{final_video.duration:.2f}s"
        )

        # 3. CHECK DURATION
        if not (
            MIN_CLIP_DURATION
            <= final_video.duration
            <= MAX_CLIP_DURATION
        ):

            print(
                f"[WARNING] Duration {final_video.duration:.2f}s "
                f"is outside the configured "
                f"{MIN_CLIP_DURATION}-{MAX_CLIP_DURATION}s range."
            )

        # 4. ADD STICKERS
        sticker_clips = create_stickers(
            final_video.duration
        )

        # 5. ADD STICKERS TO VIDEO
        if sticker_clips:

            final_video = CompositeVideoClip(
                [final_video] + sticker_clips,
                size=(WIDTH, HEIGHT)
            ).with_audio(final_video.audio)

        # 6. ADD CAPTIONS
        captioned_video = add_captions(
            final_video,
            output_name
        )

        # 7. ADD POP SOUND EFFECTS
        sound_clips = create_pop_sounds(
            captioned_video.duration
        )

        if sound_clips and captioned_video.audio is not None:

            combined_audio = CompositeAudioClip(
                [captioned_video.audio] + sound_clips
            )

            captioned_video = captioned_video.with_audio(
                combined_audio
            )

        # 8. EXPORT
        print("\n[EXPORT] Rendering final Short...")

        captioned_video.write_videofile(
            str(output_path),
            fps=FPS,
            codec=VIDEO_CODEC,
            audio_codec=AUDIO_CODEC,
            preset=EXPORT_PRESET,
            threads=4,
            logger="bar"
        )

        print("\n" + "=" * 65)
        print(f"[SUCCESS] {output_path}")
        print(f"[DURATION] {captioned_video.duration:.2f}s")
        print("=" * 65)

        return output_path

    except Exception as error:

        print(f"[MOMENT ERROR] {moment_id}: {error}")
        traceback.print_exc()

        return None

    finally:

        # Close generated resources.
        for clip in sound_clips:
            try:
                clip.close()
            except Exception:
                pass

        for clip in sticker_clips:
            try:
                clip.close()
            except Exception:
                pass

        for clip in source_clips:
            try:
                clip.close()
            except Exception:
                pass

        for clip in [captioned_video, final_video]:
            if clip is not None:
                try:
                    clip.close()
                except Exception:
                    pass

# ============================================================
# MAIN
# ============================================================

def main(json_filename):

    print("\n" + "=" * 65)
    print("             COLCASES SHORTS CLIP GENERATOR")
    print("=" * 65)

    json_path = Path(json_filename)

    # Relative paths are resolved from the project root.
    if not json_path.is_absolute():
        json_path = BASE_DIR / json_path

    if not json_path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {json_path}"
        )

    print(f"[JSON] {json_path}")

    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    video_name = data.get("video")

    if not video_name:
        raise ValueError("JSON does not contain a video filename.")

    source_path = find_video(video_name)

    print(f"[VIDEO] {source_path}")

    moments = data.get("moments", [])

    print(f"[MOMENTS] {len(moments)}")

    if not moments:
        print("[ERROR] No moments found in JSON.")
        return

    source_video = VideoFileClip(str(source_path))

    successful = []
    failed = []

    try:
        for moment in moments:

            result = generate_moment(
                source_video,
                moment
            )

            if result:
                successful.append(result)
            else:
                failed.append(
                    moment.get("moment_id", "UNKNOWN")
                )

    finally:
        source_video.close()

    print("\n" + "=" * 65)
    print("                  GENERATION SUMMARY")
    print("=" * 65)

    print(f"Total moments: {len(moments)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    for path in successful:
        print(f"[CREATED] {path.name}")

    for moment_id in failed:
        print(f"[FAILED] {moment_id}")

    print("=" * 65)


if __name__ == "__main__":
    main("results/Her Killer Was Finally Caught at 84.json")
