import json, re
from pathlib import Path
from faster_whisper import WhisperModel
from llama_cpp import Llama

# CONFIG
BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_DIR = BASE_DIR / "videos"
RESULTS_DIR = BASE_DIR / "results"
MODEL_PATH = BASE_DIR / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"

WHISPER_MODEL = "base"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
CHUNK_SECONDS = 150
OVERLAP_SECONDS = 25
MIN_SOURCE_CLIP = 5
MAX_SOURCE_CLIP = 25
MIN_SHORT_DURATION = 30
MAX_SHORT_DURATION = 60
MAX_CANDIDATES = 50
MAX_CLIPS_PER_MOMENT = 5
MAX_HIGHLIGHTS = 15

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# LOAD MODELS
whisper = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
llm = Llama(model_path=str(MODEL_PATH), n_ctx=8192, n_threads=4, verbose=False)


def normalize_word(word):
    return re.sub(r"[^A-Z0-9']", "", str(word).upper())


def extract_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Qwen did not return valid JSON.")
    return json.loads(match.group())


def ask_qwen(prompt, temperature=0.3):
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are an expert documentary video editor. Return only valid JSON. Never invent facts."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        response_format={"type": "json_object"}
    )
    return extract_json(response["choices"][0]["message"]["content"])


def transcribe_video(video_path):
    segments, _ = whisper.transcribe(str(video_path), word_timestamps=True, vad_filter=True, beam_size=5)
    return [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segments if s.text.strip()]


def split_chunks(segments, video_duration):
    chunks, start = [], 0
    while start < video_duration:
        end = min(start + CHUNK_SECONDS, video_duration)
        chunk = [s for s in segments if s["start"] < end and s["end"] > start]
        if chunk:
            chunks.append({"start": start, "end": end, "segments": chunk})
        start += CHUNK_SECONDS - OVERLAP_SECONDS
    return chunks


def get_transcript_between(segments, start, end):
    return " ".join(s["text"] for s in segments if s["start"] < end and s["end"] > start)


def find_candidate_clips(chunk, video_name):
    transcript = "\n".join(f'[{s["start"]:.2f}-{s["end"]:.2f}] {s["text"]}' for s in chunk["segments"])

    prompt = f"""
You are an expert editor for high-quality documentary YouTube Shorts.

VIDEO: {video_name}
CHUNK START: {chunk["start"]}
CHUNK END: {chunk["end"]}

TRANSCRIPT:
{transcript}

Find individual SOURCE CLIPS that could be combined with other clips to create compelling documentary Shorts.

Find:
- Mysterious events and unexplained incidents.
- Important discoveries and revelations.
- Interesting historical facts.
- Dangerous situations and survival stories.
- Emotional human experiences.
- Investigations, evidence and unexpected developments.
- Strong hooks and memorable endings.

RULES:
1. Each source clip must be 5-25 seconds long.
2. Use GLOBAL timestamps from the original video.
3. Each clip should communicate one meaningful piece of information.
4. Prefer complete sentences and natural speech boundaries.
5. Clips do not need to make sense independently.
6. Do not invent facts or change the original meaning.
7. Do not select ordinary explanations without storytelling value.
8. Find different parts of the story, not several copies of the same moment.
9. Include clips that could work as hooks, context, escalation, revelations or endings.
10. A clip can be selected even if its related clip occurs elsewhere in the documentary.

Return JSON:
{{
  "clips": [
    {{
      "start": 120.5,
      "end": 135.5,
      "topic": "A mysterious disappearance",
      "role": "Hook",
      "reason": "Immediately creates curiosity"
    }}
  ]
}}

Return no more than 5 strong candidates.
"""

    data = ask_qwen(prompt)
    candidates = []

    for item in data.get("clips", []):
        try:
            start, end = float(item["start"]), float(item["end"])
            duration = end - start

            if not MIN_SOURCE_CLIP <= duration <= MAX_SOURCE_CLIP:
                continue
            if start < chunk["start"] or end > chunk["end"]:
                continue

            candidates.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(duration, 2),
                "topic": str(item.get("topic", "")),
                "role": str(item.get("role", "Story")),
                "reason": str(item.get("reason", "")),
                "transcript": get_transcript_between(chunk["segments"], start, end)
            })
        except (KeyError, TypeError, ValueError):
            continue

    return candidates


def remove_duplicate_candidates(candidates):
    candidates.sort(key=lambda x: x["start"])
    unique = []

    for clip in candidates:
        duplicate = False
        for existing in unique:
            overlap = max(0, min(clip["end"], existing["end"]) - max(clip["start"], existing["start"]))
            shorter = min(clip["duration"], existing["duration"])

            if shorter and overlap / shorter > 0.65:
                duplicate = True
                break

        if not duplicate:
            unique.append(clip)

    return unique


def group_clips_into_stories(candidates, video_name):
    if len(candidates) < 2:
        return []

    candidate_text = json.dumps([
        {
            "clip_id": f"C{i+1:03d}",
            "start": c["start"],
            "end": c["end"],
            "duration": c["duration"],
            "topic": c["topic"],
            "role": c["role"],
            "transcript": c["transcript"]
        }
        for i, c in enumerate(candidates)
    ], ensure_ascii=False)

    prompt = f"""
You are a professional documentary short-form story editor.

VIDEO: {video_name}

You have been given individual source clips from a long documentary.

Your job is to combine related source clips into complete, engaging 30-60 second YouTube Shorts.

SOURCE CLIPS:
{candidate_text}

IMPORTANT:
A single Short can contain multiple source clips from different parts of the original video.

STORY ASSEMBLY RULES:
1. Each moment must contain 2-{MAX_CLIPS_PER_MOMENT} source clips.
2. Combined duration must be 30-60 seconds.
3. Clips must tell ONE coherent story.
4. Do not combine unrelated topics just to reach the duration.
5. Build a meaningful sequence: Hook, Context, Escalation, Revelation, Ending.
6. Use the strongest clip as the hook when appropriate.
7. A clip can be used in only ONE moment.
8. Prefer original chronological order unless a different order is essential for clarity.
9. Preserve the original timestamps.
10. Do not invent narration or events.
11. Every selected clip must have a clear storytelling purpose.
12. Avoid repetitive information.
13. It is acceptable to leave unused clips out.
14. If no coherent combination exists, return an empty moments array.

Return ONLY JSON:
{{
  "moments": [
    {{
      "title": "The Mystery Nobody Could Solve",
      "story_type": "Cold Case",
      "hook": "A mysterious opening line.",
      "narrative": "A short explanation of the combined story.",
      "ending": "Revelation",
      "clip_ids": ["C001", "C004", "C009"],
      "highlight_words": ["MYSTERY", "DISCOVERED", "NEVER"],
      "reason": "Why these clips make a compelling Short."
    }}
  ]
}}
"""

    data = ask_qwen(prompt, temperature=0.2)
    return data.get("moments", [])


def validate_highlights(words, clips):
    spoken = {
        normalize_word(w)
        for clip in clips
        for w in clip["transcript"].split()
    }

    result = []
    for word in words:
        word = normalize_word(word)
        if word and word in spoken and word not in result:
            result.append(word)
        if len(result) >= MAX_HIGHLIGHTS:
            break

    return result


def build_moments(stories, candidates):
    candidate_map = {f"C{i+1:03d}": c for i, c in enumerate(candidates)}
    moments, used_clips = [], set()

    for index, story in enumerate(stories, 1):
        selected = []
        for clip_id in story.get("clip_ids", []):
            clip = candidate_map.get(clip_id)
            if clip and clip_id not in used_clips:
                selected.append((clip_id, clip))

        if not 2 <= len(selected) <= MAX_CLIPS_PER_MOMENT:
            continue

        total_duration = sum(clip["duration"] for _, clip in selected)
        if not MIN_SHORT_DURATION <= total_duration <= MAX_SHORT_DURATION:
            continue

        # Preserve original video chronology.
        selected.sort(key=lambda x: x[1]["start"])

        clips = []
        for clip_id, clip in selected:
            clips.append({
                "clip_id": clip_id,
                "start": clip["start"],
                "end": clip["end"],
                "duration": clip["duration"],
                "role": clip["role"],
                "transcript": clip["transcript"]
            })

        highlights = validate_highlights(story.get("highlight_words", []), [c for _, c in selected])

        moments.append({
            "moment_id": f"M{index:03d}",
            "title": str(story.get("title", "Untitled Documentary Story")),
            "story_type": str(story.get("story_type", "Documentary")),
            "hook": str(story.get("hook", "")),
            "narrative": str(story.get("narrative", "")),
            "ending": str(story.get("ending", "")),
            "reason": str(story.get("reason", "")),
            "total_duration": round(total_duration, 2),
            "highlight_words": highlights,
            "clips": clips
        })

        used_clips.update(clip_id for clip_id, _ in selected)

    return moments


def get_video_duration(video_path):
    from moviepy import VideoFileClip
    with VideoFileClip(str(video_path)) as video:
        return video.duration


def main(file_name):
    video_path = Path(file_name)
    if not video_path.is_absolute():
        video_path = VIDEO_DIR / video_path

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Qwen model not found: {MODEL_PATH}")

    print(f"\nAnalyzing documentary: {video_path.name}")
    video_duration = get_video_duration(video_path)

    print("Transcribing video...")
    segments = transcribe_video(video_path)
    if not segments:
        print("No speech detected.")
        return

    chunks = split_chunks(segments, video_duration)
    candidates = []

    print(f"Found {len(chunks)} transcript chunks.")

    for index, chunk in enumerate(chunks, 1):
        print(f"Finding source clips: {index}/{len(chunks)}")
        try:
            candidates.extend(find_candidate_clips(chunk, video_path.name))
        except Exception as error:
            print(f"Chunk {index} failed: {error}")

    candidates = remove_duplicate_candidates(candidates)
    candidates = candidates[:MAX_CANDIDATES]

    print(f"\nUnique source clips found: {len(candidates)}")
    print("Grouping clips into documentary stories...")

    stories = group_clips_into_stories(candidates, video_path.name)
    moments = build_moments(stories, candidates)

    output = {
        "video": video_path.name,
        "video_duration": round(video_duration, 2),
        "total_candidates": len(candidates),
        "total_moments": len(moments),
        "total_source_clips_used": sum(len(m["clips"]) for m in moments),
        "highlight_words": list(dict.fromkeys(w for m in moments for w in m["highlight_words"])),
        "moments": moments
    }

    output_path = RESULTS_DIR / f"{video_path.stem}_clips.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print("\nDocumentary analysis complete!")
    print(f"Candidates: {len(candidates)}")
    print(f"Story moments: {len(moments)}")
    print(f"Source clips used: {output['total_source_clips_used']}")
    print(f"Saved: {output_path}")

def close_models():
    global llm, whisper
    if llm is not None:
        llm.close()
        llm = None

if __name__ == "__main__":
    try:
     main("WISCONSIN 1974 Cold Case Solved After 50 Years — Her Killer Was Finally Caught at 84.mp4")
    finally:
        close_models()