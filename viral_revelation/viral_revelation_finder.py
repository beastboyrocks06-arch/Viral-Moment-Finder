
import json, re, traceback, argparse
from pathlib import Path

from faster_whisper import WhisperModel
from llama_cpp import Llama
from json_repair import repair_json

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_DIR = BASE_DIR / "videos"
RESULTS_DIR = BASE_DIR / "results" / "viral-revelation"
MODEL_PATH = BASE_DIR / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"

VIDEO_TYPE = "PODCAST"

WHISPER_MODEL = "base"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

CHUNK_SECONDS = 150
OVERLAP_SECONDS = 25

MIN_CLIP_DURATION = 30
MAX_CLIP_DURATION = 60

MAX_MOMENTS_PER_CHUNK = 5
MAX_HIGHLIGHTS_PER_CHUNK = 12

LLM_THREADS = 4
LLM_CONTEXT_SIZE = 8192

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD MODELS
# ============================================================

print("[MODEL] Loading Whisper...")
whisper = WhisperModel(
    WHISPER_MODEL,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)

print("[MODEL] Loading Qwen...")
llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=LLM_CONTEXT_SIZE,
    n_threads=LLM_THREADS,
    verbose=False
)

# ============================================================
# HELPERS
# ============================================================

def normalize_word(word):
    return re.sub(r"[^A-Z0-9']", "", str(word).upper())


def extract_json(text):
    text = text.strip()
    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text,
        flags=re.MULTILINE
    ).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = repair_json(text, return_objects=True)

    if not isinstance(data, dict):
        raise ValueError("Qwen did not return a valid JSON object.")

    return data


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
    step = CHUNK_SECONDS - OVERLAP_SECONDS

    while start < video_duration:
        end = min(start + CHUNK_SECONDS, video_duration)

        chunk_segments = [
            segment for segment in segments
            if segment["start"] < end
            and segment["end"] > start
        ]

        if chunk_segments:
            chunks.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "segments": chunk_segments
            })

        start += step

    return chunks


def validate_highlight_words(words, transcript):
    valid_words = set()

    for segment in transcript:
        for word in segment["text"].split():
            normalized = normalize_word(word)

            if normalized:
                valid_words.add(normalized)

    result = []

    if not isinstance(words, list):
        return result

    for word in words:
        word = normalize_word(word)

        if word and word in valid_words and word not in result:
            result.append(word)

        if len(result) >= MAX_HIGHLIGHTS_PER_CHUNK:
            break

    return result


# ============================================================
# REVELATION / VIRAL MOMENT ANALYSIS
# ============================================================

def analyze_chunk(chunk, video_name):
    transcript_text = "\n".join(
        f'[{segment["start"]:.2f}-{segment["end"]:.2f}] {segment["text"]}'
        for segment in chunk["segments"]
    )

    prompt = f"""
You are an elite YouTube Shorts editor, viral content strategist
and professional podcast/interview clip selector.

Your task is to find the MOST INTERESTING, SURPRISING,
THOUGHT-PROVOKING and SHAREABLE COMPLETE VIDEO SEQUENCES.

CHANNEL: RevelationClips
VIDEO: {video_name}

CHUNK START: {chunk["start"]}
CHUNK END: {chunk["end"]}
CHUNK DURATION: {chunk["end"] - chunk["start"]}

TRANSCRIPT:
{transcript_text}

==================================================
1. CONTENT DISCOVERY
==================================================

Find interesting moments from ANY type of content:

PODCASTS AND INTERVIEWS:
- Unexpected confessions.
- Surprising personal experiences.
- Controversial opinions.
- Interesting disagreements.
- Unbelievable real-life stories.
- Unexpected answers.
- Secrets or revelations.
- Strong opinions and debates.
- Life lessons and powerful insights.
- Incredible achievements.
- Unusual experiences.

CHALLENGES AND ENTERTAINMENT:
- Impossible challenges.
- Unexpected results.
- Expensive experiences.
- Incredible demonstrations.
- Funny reactions.
- Unexpected twists.
- Extreme situations.
- Unbelievable comparisons.
- Surprising experiments.

FACTS AND DISCOVERIES:
- Mind-blowing facts.
- Scientific discoveries.
- Technology revelations.
- Historical facts.
- Hidden details.
- Interesting explanations.
- Things people commonly misunderstand.
- Unexpected connections.

HUMAN STORIES:
- Emotional moments.
- Personal struggles.
- Life-changing experiences.
- Motivational stories.
- Unexpected decisions.
- Moments of courage.
- Incredible survival stories.

Do not restrict yourself to any single category.

==================================================
2. WHAT MAKES A GREAT REVELATIONCLIPS MOMENT?
==================================================

Prioritize moments that make viewers think:

- "Wait, WHAT?"
- "I never knew that!"
- "How is that even possible?"
- "I need to hear the rest of this."
- "That completely changes how I see things."
- "I can't believe this actually happened."

Look for:

1. A strong opening statement or intriguing question.
2. An interesting explanation or story.
3. A surprising detail, revelation or payoff.

A moment should feel like a complete mini-story,
not just a random sentence taken from a conversation.

==================================================
3. CLIP SELECTION RULES
==================================================

Identify up to {MAX_MOMENTS_PER_CHUNK} strong opportunities.

Every selected clip MUST:

- Be between {MIN_CLIP_DURATION} and {MAX_CLIP_DURATION} seconds.
- Be understandable without watching the original video.
- Have a compelling opening.
- Contain meaningful progression.
- Have a satisfying conclusion or interesting takeaway.
- Give viewers a reason to continue watching.

Avoid:
- Greetings and introductions.
- Ordinary small talk.
- Repetitive statements.
- Long pauses and irrelevant discussion.
- Incomplete explanations.
- Contextless sentences.
- Weak reactions without a meaningful event.
- Multiple clips covering the same moment.

Do not create clips simply to fill the quota.

QUALITY IS MORE IMPORTANT THAN QUANTITY.

==================================================
4. TIMESTAMP REQUIREMENTS
==================================================

All timestamps MUST be absolute seconds from the ORIGINAL VIDEO.

Never return chunk-relative timestamps.

For every clip:

- start >= {chunk["start"]}
- end <= {chunk["end"]}
- start >= 0
- end > start
- duration >= {MIN_CLIP_DURATION}
- duration <= {MAX_CLIP_DURATION}

Use timestamps from the supplied transcript.

Never guess timestamps.
Never invent timestamps.
Never extend a clip with unrelated content.

Find the natural beginning of the conversation or story
and its natural ending.

==================================================
5. CONTENT QUALITY
==================================================

Prefer:
- Surprising revelations.
- Unexpected confessions.
- Interesting facts.
- Strong opinions.
- Funny and unexpected reactions.
- Powerful explanations.
- Emotional stories.
- Incredible demonstrations.
- Unusual experiences.
- Memorable conversations.

Do not confuse loudness with interesting content.

A calm explanation of something extraordinary can be
more valuable than a loud but meaningless reaction.

Do not invent facts, dialogue, names, prices or events.

When a speaker expresses an opinion, do not present it
as an established fact.

==================================================
6. TITLE REQUIREMENTS
==================================================

Write a catchy, natural YouTube Shorts title.

Titles should:
- Create curiosity.
- Be specific to the actual moment.
- Sound human-written.
- Prefer 35-65 characters.
- Use Title Case for important words.
- Include 1-2 relevant emojis when appropriate.

Examples:

- "He Revealed Something Nobody Expected! 😳"
- "This Completely Changes Everything! 🤯"
- "Nobody Was Ready For His Answer! 😭"
- "The Truth Behind This Is Insane! 😱"
- "He Explained Something Mind-Blowing! 🤯"
- "This Is Something Most People Never Notice!"
- "The Most Unexpected Answer Ever! 😂"
- "What He Said Left Everyone Speechless! 😳"

Never use misleading clickbait.

==================================================
7. REASON AND HOOK
==================================================

reason:
Explain specifically why the moment is interesting.

hook:
Use a compelling actual line from the transcript whenever possible.

Never invent dialogue or quotations.

==================================================
8. SCORING
==================================================

Score each moment from 0 to 100.

Use this exact breakdown:

hook: 0-25
surprise: 0-25
emotion: 0-20
clarity: 0-15
shorts_potential: 0-15

Total score MUST equal the sum of all five categories.

Do not assign identical scores to every moment.

==================================================
9. HIGHLIGHT WORDS
==================================================

Select up to {MAX_HIGHLIGHTS_PER_CHUNK} powerful words.

Prefer:
- Surprising words.
- Emotional words.
- Important names or concepts.
- Strong reactions.
- Words that emphasize the revelation.

Every word MUST actually appear in the transcript.

Do not invent words.

==================================================
10. POSTED STATUS
==================================================

Every moment MUST contain:

"posted": false

==================================================
11. FINAL VALIDATION
==================================================

Verify every moment:

- Numeric start and end.
- Absolute video timestamps.
- Inside chunk boundaries.
- Duration between {MIN_CLIP_DURATION} and {MAX_CLIP_DURATION}.
- Complete and interesting sequence.
- Accurate title.
- Reason and hook supported by transcript.
- Score between 0 and 100.
- Score equals breakdown total.
- posted is false.

Discard candidates that fail these requirements.

==================================================
12. REQUIRED JSON OUTPUT
==================================================

Return ONLY valid JSON.

Use exactly this structure:

{{
    "moments": [
        {{
            "start": 125.5,
            "end": 170.5,
            "title": "He Revealed Something Nobody Expected! 😳",
            "reason": "An unexpected revelation makes this conversation surprising and engaging.",
            "hook": "I never told anyone this before...",
            "score": 91,
            "score_breakdown": {{
                "hook": 24,
                "surprise": 23,
                "emotion": 18,
                "clarity": 13,
                "shorts_potential": 13
            }},
            "posted": false
        }}
    ],
    "highlight_words": ["NEVER", "SECRET", "IMPOSSIBLE"]
}}

If no suitable moments exist:

{{
    "moments": [],
    "highlight_words": []
}}

Remember:
Do not invent content.
Do not return isolated sentences.
Find complete, engaging sequences.
QUALITY IS MORE IMPORTANT THAN QUANTITY.
"""

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert viral Shorts editor. "
                    "Return only a valid JSON object."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
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
    moments = data.get("moments", [])

    if not isinstance(moments, list):
        raise ValueError("Invalid moments list returned by Qwen.")

    highlights = validate_highlight_words(
        data.get("highlight_words", []),
        chunk["segments"]
    )

    print(f"QWEN GENERATED MOMENTS: {len(moments)}")

    return moments, highlights


# ============================================================
# VALIDATE MOMENTS
# ============================================================

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
                    f"[REJECTED] {start:.2f}-{end:.2f} | "
                    f"{duration:.2f}s | {reason}"
                )
                continue

            breakdown = moment.get("score_breakdown", {})

            if not isinstance(breakdown, dict):
                breakdown = {}

            score_limits = {
                "hook": 25,
                "surprise": 25,
                "emotion": 20,
                "clarity": 15,
                "shorts_potential": 15
            }

            clean_breakdown = {}

            for key, maximum in score_limits.items():
                value = float(breakdown.get(key, 0))
                clean_breakdown[key] = max(
                    0,
                    min(maximum, round(value))
                )

            score = sum(clean_breakdown.values())

            valid.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(duration, 2),
                "title": str(moment.get("title", "Untitled")),
                "reason": str(moment.get("reason", "")),
                "hook": str(moment.get("hook", "")),
                "score": score,
                "score_breakdown": clean_breakdown,
                "highlight_words": [],
                "posted": False
            })

        except (KeyError, TypeError, ValueError) as error:
            rejected += 1
            print(f"[INVALID MOMENT] {error}: {moment}")

    print(f"Valid moments: {len(valid)}, Rejected: {rejected}")
    return valid


# ============================================================
# ATTACH HIGHLIGHTS
# ============================================================

def attach_moment_highlights(moment, segments, all_highlights):
    moment_words = set()

    for segment in segments:
        if (
            segment["start"] < moment["end"]
            and segment["end"] > moment["start"]
        ):
            for word in segment["text"].split():
                normalized = normalize_word(word)

                if normalized:
                    moment_words.add(normalized)

    moment["highlight_words"] = [
        word for word in all_highlights
        if word in moment_words
    ]

    return moment


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicate_moments(moments):
    moments = sorted(
        moments,
        key=lambda item: item.get("score", 0),
        reverse=True
    )

    unique = []

    for moment in moments:
        duplicate = False

        for existing in unique:
            overlap = max(
                0,
                min(moment["end"], existing["end"])
                - max(moment["start"], existing["start"])
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

    return sorted(unique, key=lambda item: item["start"])


# ============================================================
# VIDEO DURATION
# ============================================================

def get_video_duration(video_path):
    from moviepy import VideoFileClip

    with VideoFileClip(str(video_path)) as video:
        return video.duration


# ============================================================
# FORMAT FOR EXISTING CLIP GENERATOR
# ============================================================

def format_moment(moment, index):
    return {
        "moment_id": f"moment_{index:02d}",
        "title": moment["title"],
        "reason": moment["reason"],
        "hook": moment["hook"],
        "score": moment["score"],
        "score_breakdown": moment["score_breakdown"],
        "highlight_words": moment["highlight_words"],
        "duration": moment["duration"],
        "posted": False,
        "clips": [
            {
                "clip_id": f"clip_{index:02d}",
                "start": moment["start"],
                "end": moment["end"]
            }
        ]
    }


# ============================================================
# MAIN
# ============================================================

def main(file_name):
    video_path = Path(file_name)

    if not video_path.is_absolute():
        video_path = VIDEO_DIR / video_path

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"\nAnalyzing: {video_path.name}")
    print(f"Channel: RevelationClips")
    print(f"Video type: {VIDEO_TYPE}")

    video_duration = get_video_duration(video_path)

    print("Transcribing video...")
    segments = transcribe_video(video_path)

    if not segments:
        print("[WARNING] No speech detected.")
        return

    chunks = split_chunks(segments, video_duration)

    all_moments = []
    all_highlights = []

    print(f"Found {len(chunks)} transcript chunks.")

    for index, chunk in enumerate(chunks, start=1):
        print(f"\nAnalyzing chunk {index}/{len(chunks)}...")

        for attempt in range(2):
            try:
                moments, highlights = analyze_chunk(
                    chunk,
                    video_path.name
                )

                validated = validate_moments(
                    moments,
                    chunk,
                    video_duration
                )

                all_moments.extend(validated)
                all_highlights.extend(highlights)

                break

            except Exception as error:
                print(
                    f"Chunk {index}, attempt {attempt + 1} failed: {error}"
                )

                if attempt == 1:
                    traceback.print_exc()

    all_highlights = list(dict.fromkeys(all_highlights))
    all_moments = remove_duplicate_moments(all_moments)

    for moment in all_moments:
        attach_moment_highlights(
            moment,
            segments,
            all_highlights
        )

    formatted_moments = [
        format_moment(moment, index)
        for index, moment in enumerate(all_moments, start=1)
    ]

    output = {
        "video": video_path.name,
        "video_type": VIDEO_TYPE,
        "video_duration": round(video_duration, 2),
        "total_moments": len(formatted_moments),
        "highlight_words": all_highlights,
        "moments": formatted_moments
    }

    output_path = RESULTS_DIR / f"{video_path.stem}.json"

    temp_path = output_path.with_suffix(".json.tmp")

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    temp_path.replace(output_path)

    print("\nAnalysis complete!")
    print(f"Channel: RevelationClips")
    print(f"Moments found: {len(formatted_moments)}")
    print(f"Highlight words: {len(all_highlights)}")
    print(f"Saved: {output_path.resolve()}")


# ============================================================
# CLEANUP
# ============================================================

def close_models():
    global llm, whisper

    if llm is not None:
        try:
            llm.close()
        except Exception:
            pass

        llm = None

    whisper = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True,
        help="Video filename inside videos/"
    )

    args = parser.parse_args()

    try:
        main(args.video)
    finally:
        close_models()
