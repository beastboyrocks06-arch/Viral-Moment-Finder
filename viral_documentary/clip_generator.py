
import os, sys, json, random, re, traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

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
STICKER_DIR = BASE_DIR / "sticker"
SFX_DIR = BASE_DIR / "sfx"

CHANNEL = os.getenv("CHANNEL", "DOCUMENTARY_MOMENT").upper()

CHANNEL_MAP = {
    "VIRAL_MOMENT": "viral-moment",
    "DOCUMENTARY_MOMENT": "documentary-moment",
}

OUTPUT_DIR = BASE_DIR / "clips" / CHANNEL_MAP.get(
    CHANNEL, "documentary-moment"
)

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

MAX_MOMENTS_PER_RUN = 2

VIDEO_CODEC = "libx264"
AUDIO_CODEC = "aac"
EXPORT_PRESET = "medium"


# ============================================================
# LAYOUT SETTINGS
# ============================================================

DOCUMENTARY_LAYOUT = "crop"
PODCAST_LAYOUT = "blurred_fit"
GENERAL_LAYOUT = "blurred_fit"


# ============================================================
# CAPTION SETTINGS
# ============================================================

ENABLE_CAPTIONS = True
CAPTION_MODULE = None


# ============================================================
# STICKER SETTINGS
# ============================================================

ENABLE_STICKERS = False

STICKER_TARGET_WIDTH = 350
STICKER_MIN_DURATION = 1.5
STICKER_MAX_DURATION = 2.5
MAX_STICKERS_PER_MOMENT = 2


# ============================================================
# SOUND EFFECT SETTINGS
# ============================================================

ENABLE_POP_SOUND = True
MAX_POP_SOUNDS = 2
POP_VOLUME = 0.25


# ============================================================
# CONTENT PROFILES
# ============================================================

PROFILES = {
    "DOCUMENTARY": {
        "layout": DOCUMENTARY_LAYOUT,
        "stickers": False,
        "sound_effects": ENABLE_POP_SOUND,
    },
    "PODCAST": {
        "layout": PODCAST_LAYOUT,
        "stickers": False,
        "sound_effects": ENABLE_POP_SOUND,
    },
    "GENERAL": {
        "layout": GENERAL_LAYOUT,
        "stickers": False,
        "sound_effects": ENABLE_POP_SOUND,
    },
}


# ============================================================
# LOAD CAPTION GENERATOR
# ============================================================

def load_caption_generator():
    import importlib.util

    possible_paths = [
        BASE_DIR / "caption_generator.py",
        BASE_DIR.parent / "caption_generator.py",
        BASE_DIR.parent.parent / "caption_generator.py",
    ]

    for path in possible_paths:
        if not path.is_file():
            continue

        try:
            if str(path.parent) not in sys.path:
                sys.path.insert(0, str(path.parent))

            spec = importlib.util.spec_from_file_location(
                "shared_caption_generator",
                path
            )

            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "generate_caption_overlay"):
                print(f"[CAPTION] Loaded: {path}")
                return module

        except Exception as error:
            print(f"[CAPTION] Could not load {path}: {error}")

    print("[CAPTION] No compatible caption generator found.")
    return None


if ENABLE_CAPTIONS:
    CAPTION_MODULE = load_caption_generator()


# ============================================================
# HELPERS
# ============================================================

def safe_filename(value):
    value = re.sub(r'[<>:"/\\|?*]', "", str(value))
    value = re.sub(r"\s+", "_", value.strip())
    return value[:150]


def find_video(video_name):
    video_path = Path(video_name)

    if video_path.is_absolute() and video_path.is_file():
        return video_path

    possible_paths = [
        VIDEO_DIR / video_name,
        BASE_DIR / video_name,
    ]

    for path in possible_paths:
        if path.is_file():
            return path

    matches = list(BASE_DIR.rglob(Path(video_name).name))

    for path in matches:
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"Could not find source video: {video_name}"
    )


def get_video_type(data):
    video_type = str(
        data.get("video_type", "GENERAL")
    ).upper().strip()

    if video_type not in PROFILES:
        print(
            f"[WARNING] Unknown video type '{video_type}'. "
            "Using GENERAL profile."
        )
        video_type = "GENERAL"

    return video_type


def get_moment_id(moment, index):
    return str(
        moment.get("moment_id", f"moment_{index:02d}")
    )


# ============================================================
# VIDEO LAYOUT
# ============================================================

def make_vertical(clip, layout):
    original_w, original_h = clip.size

    if original_w <= 0 or original_h <= 0:
        raise ValueError("Invalid source video dimensions.")

    if layout == "crop":
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

    if layout == "blurred_fit":
        frame = clip.get_frame(0)

        background = Image.fromarray(
            frame.astype(np.uint8)
        ).convert("RGB")

        background = ImageOps.fit(
            background,
            (WIDTH, HEIGHT),
            method=Image.Resampling.LANCZOS
        )

        background = background.filter(
            ImageFilter.GaussianBlur(radius=35)
        )

        background_clip = (
            ImageClip(np.array(background))
            .with_duration(clip.duration)
        )

        scale = min(
            WIDTH / original_w,
            HEIGHT / original_h
        )

        foreground = clip.resized(scale)

        return (
            CompositeVideoClip(
                [
                    background_clip,
                    foreground.with_position("center")
                ],
                size=(WIDTH, HEIGHT)
            )
            .with_duration(clip.duration)
            .with_audio(clip.audio)
        )

    raise ValueError(f"Unknown video layout: {layout}")


# ============================================================
# EXTRACT SOURCE SEGMENTS
# ============================================================

def extract_moment_clips(source_video, moment, profile):
    clips = []

    source_segments = moment.get("clips", [])
    moment_id = moment.get("moment_id", "UNKNOWN")

    if not isinstance(source_segments, list):
        print(f"[SKIP] Invalid clips list: {moment_id}")
        return clips

    source_segments = sorted(
        source_segments,
        key=lambda item: float(item.get("start", 0))
    )

    for segment in source_segments:
        source_clip = None
        vertical_clip = None

        try:
            clip_id = segment.get("clip_id", "UNKNOWN")

            start = float(segment["start"])
            end = float(segment["end"])

            if start < 0 or end <= start:
                print(f"[SKIP] Invalid timestamps: {clip_id}")
                continue

            if end > source_video.duration:
                print(f"[SKIP] {clip_id}: End exceeds video duration.")
                continue

            duration = end - start

            print(
                f"[CUT] {clip_id} | "
                f"{start:.2f}-{end:.2f}s | "
                f"{duration:.2f}s"
            )

            source_clip = source_video.subclipped(start, end)

            vertical_clip = make_vertical(
                source_clip,
                profile["layout"]
            )

            # Keep source alive until final rendering completes.
            vertical_clip._source_reference = source_clip
            clips.append(vertical_clip)

            source_clip = None
            vertical_clip = None

        except Exception as error:
            print(f"[ERROR] Could not extract segment: {error}")
            traceback.print_exc()

        finally:
            if vertical_clip is not None:
                vertical_clip.close()

            if source_clip is not None:
                source_clip.close()

    return clips


# ============================================================
# STICKER GENERATOR
# ============================================================

def create_stickers(video_duration, profile):
    if not ENABLE_STICKERS or not profile["stickers"]:
        return []

    if not STICKER_DIR.exists():
        return []

    sticker_files = [
        *STICKER_DIR.glob("*.png"),
        *STICKER_DIR.glob("*.jpg"),
        *STICKER_DIR.glob("*.webp")
    ]

    if not sticker_files:
        return []

    random.shuffle(sticker_files)

    count = min(
        len(sticker_files),
        MAX_STICKERS_PER_MOMENT
    )

    sticker_clips = []

    for path in sticker_files[:count]:
        sticker = None

        try:
            with Image.open(path) as image:
                image = image.convert("RGBA")

                bbox = image.getchannel("A").getbbox()

                if bbox:
                    image = image.crop(bbox)

                original_w, original_h = image.size

                if original_w <= 0 or original_h <= 0:
                    continue

                scale = min(
                    STICKER_TARGET_WIDTH / original_w,
                    1.0
                )

                image = image.resize(
                    (
                        round(original_w * scale),
                        round(original_h * scale)
                    ),
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

            x = random.choice([
                45,
                WIDTH - sticker_w - 45
            ])

            y = random.choice([250, 750])

            sticker = (
                sticker
                .with_start(start)
                .with_duration(duration)
                .with_position((x, y))
            )

            sticker_clips.append(sticker)
            sticker = None

            print(f"[STICKER] {path.name} at {start:.2f}s")

        except Exception as error:
            print(f"[STICKER ERROR] {path.name}: {error}")

        finally:
            if sticker is not None:
                sticker.close()

    return sticker_clips


# ============================================================
# POP SOUND EFFECTS
# ============================================================

def create_pop_sounds(video_duration, clip_boundaries, profile):
    if not profile["sound_effects"]:
        return []

    if not POP_SFX.is_file():
        return []

    boundaries = [
        value for value in clip_boundaries
        if 0.5 <= value < video_duration - 0.5
    ]

    if not boundaries:
        return []

    if len(boundaries) > MAX_POP_SOUNDS:
        indices = np.linspace(
            0,
            len(boundaries) - 1,
            MAX_POP_SOUNDS,
            dtype=int
        )
        boundaries = [boundaries[i] for i in indices]

    sound_clips = []

    for start in boundaries:
        sound = None

        try:
            sound = AudioFileClip(str(POP_SFX))

            sound = (
                sound
                .with_start(start)
                .with_volume_scaled(POP_VOLUME)
            )

            sound_clips.append(sound)
            sound = None

            print(f"[SFX] Transition pop at {start:.2f}s")

        except Exception as error:
            print(f"[SFX ERROR] {error}")

        finally:
            if sound is not None:
                sound.close()

    return sound_clips


# ============================================================
# CAPTION GENERATION
# ============================================================

def add_captions(final_video, output_name):
    if not ENABLE_CAPTIONS or CAPTION_MODULE is None:
        print("[CAPTION] Skipping captions.")
        return final_video

    if final_video.audio is None:
        print("[CAPTION] No audio. Skipping captions.")
        return final_video

    temp_audio = OUTPUT_DIR / f"{output_name}_temp_audio.wav"

    try:
        print("[CAPTION] Extracting audio...")

        final_video.audio.write_audiofile(
            str(temp_audio),
            fps=44100,
            nbytes=2,
            codec="pcm_s16le",
            logger=None
        )

        print("[CAPTION] Generating captions...")

        caption_overlay = CAPTION_MODULE.generate_caption_overlay(
            str(temp_audio)
        )

        if caption_overlay is None:
            print("[CAPTION] No overlay returned.")
            return final_video

        composed = (
            CompositeVideoClip(
                [
                    final_video,
                    caption_overlay
                    .with_duration(final_video.duration)
                    .with_position("center")
                ],
                size=(WIDTH, HEIGHT)
            )
            .with_duration(final_video.duration)
            .with_audio(final_video.audio)
        )

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
            except OSError:
                pass


# ============================================================
# GENERATE ONE MOMENT
# ============================================================

def generate_moment(
    source_video,
    moment,
    video_type,
    source_name,
    moment_index
):
    profile = PROFILES[video_type]

    moment_id = get_moment_id(moment, moment_index)
    title = moment.get("title", moment_id)

    # Match the existing YouTube uploader naming convention.
    output_name = safe_filename(
        f"{Path(source_name).stem}_{moment_index:02d}"
    )

    output_path = OUTPUT_DIR / f"{output_name}.mp4"

    print("\n" + "=" * 65)
    print(f"[{video_type}] {moment_id}: {title}")
    print("=" * 65)

    source_clips = []
    sticker_clips = []
    sound_clips = []

    final_video = None
    captioned_video = None
    rendered_video = None

    try:
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"[SKIP] Already generated: {output_path.name}")
            return output_path

        # 1. EXTRACT SOURCE CLIPS
        source_clips = extract_moment_clips(
            source_video,
            moment,
            profile
        )

        if not source_clips:
            print("[ERROR] No valid source clips.")
            return None

        # 2. CALCULATE TRANSITION POSITIONS
        clip_boundaries = []
        elapsed = 0

        for clip in source_clips[:-1]:
            elapsed += clip.duration
            clip_boundaries.append(elapsed)

        # 3. COMBINE SOURCE CLIPS
        print("[JOIN] Combining source clips...")

        final_video = concatenate_videoclips(
            source_clips,
            method="compose"
        )

        print(f"[JOIN] Duration: {final_video.duration:.2f}s")

        # 4. VALIDATE DURATION
        if not (
            MIN_CLIP_DURATION
            <= final_video.duration
            <= MAX_CLIP_DURATION
        ):
            print(
                f"[SKIP] Duration {final_video.duration:.2f}s "
                f"is outside {MIN_CLIP_DURATION}-"
                f"{MAX_CLIP_DURATION}s."
            )
            return None

        # 5. ADD STICKERS
        sticker_clips = create_stickers(
            final_video.duration,
            profile
        )

        if sticker_clips:
            composed = (
                CompositeVideoClip(
                    [final_video] + sticker_clips,
                    size=(WIDTH, HEIGHT)
                )
                .with_duration(final_video.duration)
                .with_audio(final_video.audio)
            )

            final_video = composed

        # 6. ADD SOUND EFFECTS
        sound_clips = create_pop_sounds(
            final_video.duration,
            clip_boundaries,
            profile
        )

        if sound_clips and final_video.audio is not None:
            combined_audio = (
                CompositeAudioClip(
                    [final_video.audio] + sound_clips
                )
                .with_duration(final_video.duration)
            )

            final_video = final_video.with_audio(combined_audio)

        # 7. ADD CAPTIONS
        captioned_video = add_captions(
            final_video,
            output_name
        )

        rendered_video = captioned_video

        # 8. EXPORT
        print("\n[EXPORT] Rendering Short...")

        rendered_video.write_videofile(
            str(output_path),
            fps=FPS,
            codec=VIDEO_CODEC,
            audio_codec=AUDIO_CODEC,
            preset=EXPORT_PRESET,
            threads=4,
            logger="bar"
        )

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Export completed without a valid MP4 file.")

        print(f"\n[SUCCESS] {output_path}")
        print(f"[DURATION] {rendered_video.duration:.2f}s")

        return output_path

    except Exception as error:
        print(f"[MOMENT ERROR] {moment_id}: {error}")
        traceback.print_exc()

        if output_path.exists():
            try:
                output_path.unlink()
            except OSError:
                pass

        return None

    finally:
        resources = [
            rendered_video,
            captioned_video,
            final_video,
            *sticker_clips,
            *sound_clips,
            *source_clips
        ]

        seen = set()

        for clip in resources:
            if clip is None or id(clip) in seen:
                continue

            seen.add(id(clip))

            source_reference = getattr(
                clip,
                "_source_reference",
                None
            )

            try:
                clip.close()
            except Exception:
                pass

            if source_reference is not None:
                try:
                    source_reference.close()
                except Exception:
                    pass


# ============================================================
# MAIN
# ============================================================

def main(json_filename):
    print("\n" + "=" * 65)
    print("             UNIVERSAL SHORTS CLIP GENERATOR")
    print("=" * 65)

    json_path = Path(json_filename)

    if not json_path.is_absolute():
        json_path = BASE_DIR / json_path

    if not json_path.is_file():
        raise FileNotFoundError(
            f"JSON file not found: {json_path}"
        )

    print(f"[JSON] {json_path}")

    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object.")

    video_name = data.get("video")

    if not video_name:
        raise ValueError("JSON does not contain a video filename.")

    video_type = get_video_type(data)

    print(f"[TYPE] {video_type}")

    source_path = find_video(video_name)

    moments = data.get("moments", [])

    if not isinstance(moments, list):
        raise ValueError("JSON 'moments' must be a list.")

    print(f"[VIDEO] {source_path}")
    print(f"[MOMENTS] {len(moments)}")
    print(f"[LIMIT] Maximum moments this run: {MAX_MOMENTS_PER_RUN}")

    if not moments:
        print("[ERROR] No moments found.")
        return

    selected_moments = moments[:MAX_MOMENTS_PER_RUN]

    source_video = None
    successful = []
    failed = []

    try:
        source_video = VideoFileClip(str(source_path))

        for index, moment in enumerate(selected_moments, start=1):
            result = generate_moment(
                source_video,
                moment,
                video_type,
                source_path.name,
                index
            )

            if result:
                successful.append(result)
            else:
                failed.append(
                    get_moment_id(moment, index)
                )

    finally:
        if source_video is not None:
            source_video.close()

    print("\n" + "=" * 65)
    print("                 GENERATION SUMMARY")
    print("=" * 65)

    print(f"Content type: {video_type}")
    print(f"Total moments in JSON: {len(moments)}")
    print(f"Selected this run: {len(selected_moments)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    for path in successful:
        print(f"[CREATED] {path.name}")

    for moment_id in failed:
        print(f"[FAILED] {moment_id}")

    print("=" * 65)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            'Usage: python viral_clip_generator.py "results/video.json"'
        )
        sys.exit(1)

    main(sys.argv[1])
