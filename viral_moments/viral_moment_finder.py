import json, re, traceback
from pathlib import Path
from faster_whisper import WhisperModel
from llama_cpp import Llama

# CONFIG
BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_DIR = BASE_DIR / "videos"
RESULTS_DIR = BASE_DIR / "viral_moments" / "results"
MODEL_PATH = BASE_DIR / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"

WHISPER_MODEL = "base"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
CHUNK_SECONDS = 150
OVERLAP_SECONDS = 25
MIN_CLIP_DURATION = 30
MAX_CLIP_DURATION = 60
MAX_HIGHLIGHTS_PER_CHUNK = 12

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# LOAD MODELS
whisper = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
llm = Llama(model_path=str(MODEL_PATH), n_ctx=8192, n_threads=4, verbose=False)


def normalize_word(word):
    return re.sub(r"[^A-Z0-9']", "", str(word).upper())


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Qwen did not return valid JSON.")
    return json.loads(match.group())

def transcribe_video(video_path):
    segments, _ = whisper.transcribe(
        str(video_path),
        word_timestamps=True,
        vad_filter=True,
        beam_size=5
    )
    transcript = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            transcript.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": text
            })
    return transcript


def split_chunks(segments, video_duration):
    chunks = []
    start = 0
    while start < video_duration:
        end = min(start + CHUNK_SECONDS, video_duration)
        chunk = [s for s in segments if s["start"] < end and s["end"] > start]
        if chunk:
            chunks.append({"start": start, "end": end, "segments": chunk})
        start += CHUNK_SECONDS - OVERLAP_SECONDS
    return chunks


def validate_highlight_words(words, transcript):
    valid_words = set()
    for segment in transcript:
        valid_words.update(normalize_word(w) for w in segment["text"].split())
    result = []
    for word in words:
        word = normalize_word(word)
        if word and word in valid_words and word not in result:
            result.append(word)
        if len(result) >= MAX_HIGHLIGHTS_PER_CHUNK:
            break
    return result


def analyze_chunk(chunk, video_name):
    transcript_text = "\n".join(
        f'[{s["start"]:.2f}-{s["end"]:.2f}] {s["text"]}'
        for s in chunk["segments"]
    )

    prompt = f"""
    You are an expert YouTube Shorts editor, viral content strategist, and professional video clip selector.

    Your task is to analyze the transcript and identify the strongest COMPLETE 30-60 SECOND YouTube Shorts opportunities.

    Your goal is NOT to find individual interesting sentences.
    Your goal is to find complete, engaging video sequences that viewers would genuinely enjoy watching as standalone Shorts.

    VIDEO: {video_name}
    CHUNK START: {chunk["start"]}
    CHUNK END: {chunk["end"]}
    CHUNK DURATION: {chunk["end"] - chunk["start"]}

    TRANSCRIPT:
    {transcript_text}

    ==================================================
    1. CONTENT DISCOVERY
    ==================================================

    Identify interesting content from ANY category, including:

    - Surprising discoveries and unexpected revelations.
    - Challenges, competitions and difficult situations.
    - Expensive products, vehicles, experiences and comparisons.
    - Unusual experiences and unbelievable events.
    - Dangerous situations and close calls.
    - Funny moments and unexpected reactions.
    - Emotional reactions and personal experiences.
    - Impressive achievements and demonstrations.
    - Interesting facts and explanations.
    - Luxury experiences and extraordinary places.
    - Unexpected results and satisfying conclusions.
    - Arguments, disagreements and controversial moments.

    Do not restrict yourself to mystery, suspense, horror or documentary content.

    ==================================================
    2. CLIP SELECTION RULES
    ==================================================

    - Identify up to 5 of the strongest clipping opportunities.
    - Every selected clip MUST be between 30 and 60 seconds.
    - NEVER return a clip shorter than 30 seconds.
    - NEVER return a clip longer than 90 seconds.
    - Select complete sequences rather than individual sentences.
    - Include the setup, interesting development and payoff whenever possible.
    - Include enough surrounding dialogue to understand the event.
    - Prefer moments with a strong opening and satisfying conclusion.
    - Avoid ordinary introductions, greetings and unnecessary filler.
    - Avoid repetitive clips covering the same event.
    - Do not create clips just to fill the quota.
    - Do not invent events, dialogue, facts, reactions or prices.
    - Do not artificially extend a clip with unrelated transcript content.
    - If a topic only contains a few seconds of interesting content, DO NOT select it.
    - Return fewer clips when there are genuinely fewer suitable opportunities.

    IMPORTANT:
    A 5-second interesting sentence is NOT a valid clip.
    Find the complete 30-60 second sequence surrounding that moment, provided the surrounding content is relevant and engaging.

    For example:
    If an interesting statement occurs at 300 seconds, examine the surrounding transcript to find a natural beginning and ending for the complete sequence.

    ==================================================
    3. TIMESTAMP REQUIREMENTS
    ==================================================

    This section is extremely important.

    - All timestamps MUST be absolute seconds from the ORIGINAL VIDEO.
    - Never return timestamps relative to the chunk.
    - Use the timestamps supplied in the transcript.
    - start must represent the beginning of the selected sequence.
    - end must represent the natural conclusion of the selected sequence.
    - Every clip duration MUST satisfy:

    30 <= (end - start) <= 90

    - start MUST be greater than or equal to {chunk["start"]}.
    - end MUST be less than or equal to {chunk["end"]}.
    - Never select content outside the current chunk.
    - Never return the entire 150-second chunk as one clip.
    - Never guess timestamps.
    - Never change timestamps simply to satisfy the duration requirement.

    Before returning each moment, internally verify:
    1. Is the start timestamp inside the chunk?
    2. Is the end timestamp inside the chunk?
    3. Is the duration at least 30 seconds?
    4. Is the duration no more than 90 seconds?
    5. Does the complete sequence contain engaging content?

    If any condition fails, discard that candidate.

    ==================================================
    4. MOMENT QUALITY
    ==================================================

    Every selected moment must:

    - Make sense without watching the entire original video.
    - Have a clear subject or event.
    - Contain meaningful progression.
    - Give viewers a reason to continue watching.
    - Avoid unnecessary silence or irrelevant conversation.
    - Have a natural beginning and ending.
    - Be suitable for vertical YouTube Shorts editing.

    Prefer complete stories, discoveries, demonstrations and experiences over isolated reactions.

    ==================================================
    5. YOUTUBE TITLE REQUIREMENTS
    ==================================================

    The existing "title" field MUST contain the final YouTube-ready title.

    - Make titles engaging, curiosity-driven and accurate.
    - Ideally keep titles between 35 and 65 characters.
    - Use different titles for different moments.
    - Avoid generic titles such as "Amazing Experience".
    - Avoid misleading clickbait.
    - Do not invent information.
    - Use up to 2 relevant emojis.
    - Do not create an additional title field.

    ==================================================
    6. REASON AND HOOK
    ==================================================

    reason:
    - Briefly explain why the selected sequence is interesting.
    - Explain what makes viewers want to watch it.
    - Be specific to the actual transcript.

    hook:
    - Use an actual compelling line or moment from the transcript whenever possible.
    - Keep it short and attention-grabbing.
    - Never invent dialogue or quotations.

    ==================================================
    7. SCORING
    ==================================================

    Score every moment from 0 to 100.

    Use this exact breakdown:

    hook: 0-25
    suspense: 0-25
    emotion: 0-20
    clarity: 0-15
    shorts_potential: 0-15

    Total score MUST equal the sum of all five categories.

    Do not assign identical scores to every moment.
    Score each candidate independently based on its actual quality.

    ==================================================
    8. HIGHLIGHT WORDS
    ==================================================

    - Select up to 12 important words from the transcript.
    - Prefer powerful, emotional and attention-grabbing words.
    - Words MUST actually appear in the transcript.
    - Preserve the original words.
    - Do not invent words.
    - Return an empty array if there are no suitable highlight words.

    ==================================================
    9. POSTED STATUS
    ==================================================

    Every moment MUST contain:

    "posted": false

    ==================================================
    10. FINAL VALIDATION
    ==================================================

    Before producing JSON, verify every moment:

    - start and end are valid numbers.
    - start is inside the current chunk.
    - end is inside the current chunk.
    - duration is between 30 and 90 seconds.
    - the moment represents a complete engaging sequence.
    - title is accurate and YouTube-ready.
    - score is between 0 and 100.
    - score_breakdown contains all five categories.
    - score equals the sum of the breakdown.
    - posted is false.

    Discard any moment that fails these requirements.

    ==================================================
    11. REQUIRED JSON OUTPUT
    ==================================================

    Return ONLY valid JSON.
    Do not include Markdown.
    Do not include explanations.
    Do not include text outside the JSON object.

    Use this exact structure:

    {{
        "moments": [
            {{
                "start": 125.5,
                "end": 170.5,
                "title": "This Discovery Changed Everything! 😳",
                "reason": "A surprising revelation creates curiosity and keeps viewers interested.",
                "hook": "Nobody expected what happened next.",
                "score": 91,
                "score_breakdown": {{
                    "hook": 24,
                    "suspense": 23,
                    "emotion": 18,
                    "clarity": 13,
                    "shorts_potential": 13
                }},
                "posted": false
            }}
        ],
        "highlight_words": ["MYSTERY", "SHOCKING", "NEVER"]
    }}

    If no suitable clips exist, return:
    "moments": []

    If no suitable highlight words exist, return:
    "highlight_words": []

    Remember:
    QUALITY IS MORE IMPORTANT THAN QUANTITY.
    A complete 30-90 second sequence is mandatory.
    Never return short isolated moments.
    """

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "Return only a valid JSON object. Never use markdown."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=3000,
        response_format={"type": "json_object"}
    )

    
    choice = response["choices"][0]
    content = choice["message"]["content"]

    print(f"\nQWEN FINISH REASON: {choice.get('finish_reason')}")
    print(f"QWEN RAW RESPONSE (first 3000 chars):\n{content[:3000]}")

    data = extract_json(content)
    print(f"QWEN JSON KEYS: {list(data.keys())}")
    moments = data.get("moments", [])
    print(f"QWEN GENERATED MOMENTS: {len(moments)}")

    highlights = validate_highlight_words(
        data.get("highlight_words", []),
        chunk["segments"]
    )

    return moments, highlights


def validate_moments(moments, chunk, video_duration):
    valid = []
    rejected = 0

    for moment in moments:
        try:
            start = float(moment["start"])
            end = float(moment["end"])
            duration = end - start

            reason = None

            if start < chunk["start"] or end > chunk["end"]:
                reason = "Outside chunk boundaries"
            elif start < 0 or end > video_duration:
                reason = "Outside video boundaries"
            elif not MIN_CLIP_DURATION <= duration <= MAX_CLIP_DURATION:
                reason = f"Invalid duration: {duration:.2f}s"

            if reason:
                rejected += 1
                print(
                    f"REJECTED: {start:.2f}-{end:.2f} | "
                    f"Duration: {duration:.2f}s | Reason: {reason}"
                )
                continue

            breakdown = moment.get("score_breakdown", {})
            score = moment.get("score")

            if score is None:
                score = sum(
                    float(breakdown.get(key, 0))
                    for key in ("hook", "suspense", "emotion",
                                "clarity", "shorts_potential")
                )

            valid.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(duration, 2),
                "title": str(moment.get("title", "Untitled")),
                "reason": str(moment.get("reason", "")),
                "hook": str(moment.get("hook", "")),
                "score": max(0, min(100, round(float(score)))),
                "score_breakdown": breakdown,
                "highlight_words": [],
                "posted": False
            })

        except (KeyError, TypeError, ValueError) as error:
            rejected += 1
            print(f"INVALID MOMENT: {moment} | Error: {error}")

    print(f"Valid moments: {len(valid)}, Rejected moments: {rejected}")
    return valid

def attach_moment_highlights(moment, segments, all_highlights):
    moment_words = set()
    for segment in segments:
        if segment["start"] < moment["end"] and segment["end"] > moment["start"]:
            moment_words.update(
                normalize_word(w) for w in segment["text"].split()
            )

    moment["highlight_words"] = [
        word for word in all_highlights if word in moment_words
    ]
    return moment



def remove_duplicate_moments(moments):
    moments = sorted(
        moments,
        key=lambda x: x.get("score", 0),
        reverse=True
    )

    unique = []

    for moment in moments:
        duplicate = False

        for existing in unique:
            overlap = max(
                0,
                min(moment["end"], existing["end"]) -
                max(moment["start"], existing["start"])
            )

            shorter = min(
                moment["duration"],
                existing["duration"]
            )

            if shorter and overlap / shorter > 0.65:
                duplicate = True
                break

        if not duplicate:
            unique.append(moment)

    return sorted(unique, key=lambda x: x["start"])


def get_video_duration(video_path):
    from moviepy import VideoFileClip
    with VideoFileClip(str(video_path)) as video:
        return video.duration


from json_repair import repair_json

def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = repair_json(text, return_objects=True)
        if not isinstance(repaired, dict):
            raise ValueError("Qwen did not return a valid JSON object.")
        return repaired


def main(file_name):
    video_path = Path(file_name)
    if not video_path.is_absolute():
        video_path = VIDEO_DIR / video_path
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"\nAnalyzing: {video_path.name}")
    video_duration = get_video_duration(video_path)
    print("Transcribing video...")
    segments = transcribe_video(video_path)

    chunks = split_chunks(segments, video_duration)
    all_moments, all_highlights = [], []
    print(f"Found {len(chunks)} transcript chunks.")

        
    for index, chunk in enumerate(chunks, 1):
        print(f"Analyzing chunk {index}/{len(chunks)}...")

        for attempt in range(2):
            try:
                moments, highlights = analyze_chunk(
                    chunk,
                    video_path.name
                )

                all_highlights.extend(highlights)

                all_moments.extend(
                    validate_moments(
                        moments,
                        chunk,
                        video_duration
                    )
                )

                break

            except Exception as error:
                print(
                    f"Chunk {index}, attempt {attempt + 1} failed: {error}"
                )

                if attempt == 1:
                    traceback.print_exc()


    all_highlights = list(dict.fromkeys(all_highlights))
    all_moments = remove_duplicate_moments(all_moments)

    # Assign sequential IDs after sorting and removing duplicates
    for index, moment in enumerate(all_moments, start=1):
        moment["id"] = f"{index:02d}"

    for moment in all_moments:
        attach_moment_highlights(moment, segments, all_highlights)

    output = {
        "video": video_path.name,
        "video_duration": round(video_duration, 2),
        "total_moments": len(all_moments),
        "highlight_words": all_highlights,
        "moments": all_moments
    }

    output_path = RESULTS_DIR / f"{video_path.stem}.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print("\nAnalysis complete!")
    print(f"Moments found: {len(all_moments)}")
    print(f"Highlight words: {len(all_highlights)}")
    print(f"Saved: {output_path.resolve()}")


def close_models():
    global llm, whisper
    if llm is not None:
        llm.close()
        llm = None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Video filename inside videos/")
    args = parser.parse_args()

    main(args.video)