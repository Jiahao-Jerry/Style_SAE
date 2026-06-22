"""
Step 1 (Local) — Generate 4,900 rewrites (700 posts × 7 axes) using Claude Haiku.

Input:  2550_posts.jsonl  (from github.com/Jiahao-Jerry/SURE)
Output: sae_clean_rewrites.json  (list of 700 post entries, each with 7 axis rewrites)

Post selection: 700 posts sampled from 2550_posts.jsonl with fixed N_PER_TOPIC posts
per topic (17 topics × 41 posts = 697, rounded up to 700). Within each topic, posts
are ranked by mid-range axis score count (how many of 7 axes fall in 0.2–0.8) so
rewrites have maximum room to shift in either direction.

Cost estimate: ~4,900 rewrites × $0.00062 ≈ $3.00
Run time:      ~25 minutes
"""

import json, time, random
import numpy as np
import pandas as pd
from pathlib import Path
from anthropic import Anthropic

POSTS_FILE = "2550_posts.jsonl"

# ── Config ────────────────────────────────────────────────────────
N_PER_TOPIC    = 41            # posts per topic (17 × 41 = 697 ≈ 700)
CACHE_FILE     = "sae_clean_rewrites.json"
AXIS_NAMES     = ["reading_level", "background", "abstract_concrete",
                  "tone", "humor", "narrativity", "grounding"]
SEED           = 42

client = Anthropic()

# ── Axis definitions (current score + target in prompt) ───────────
AXIS_DEFS = {
    "reading_level": {
        "up":   "more specialist vocabulary and complex sentence structure",
        "down": "simpler vocabulary and shorter sentences, as if explaining to a general audience",
    },
    "background": {
        "up":   "assume more prior knowledge — skip context and get straight to the point",
        "down": "unpack more background context so a newcomer can follow along",
    },
    "abstract_concrete": {
        "up":   "more concrete — replace vague claims with specific named facts, events, or people",
        "down": "more abstract — remove specific examples and speak in general terms",
    },
    "tone": {
        "up":   "more measured and analytical — cool the emotional temperature down",
        "down": "more emotionally charged and urgent",
    },
    "humor": {
        "up":   "wittier and more playful — add dry humor or sarcasm",
        "down": "more earnest and serious — remove any wit or comedy",
    },
    "narrativity": {
        "up":   "more story-like — add a brief personal anecdote or scene-setting opener",
        "down": "pure assertion style — remove personal framing, just make the argument directly",
    },
    "grounding": {
        "up":   "more example-driven — add a concrete analogy or real-world comparison",
        "down": "more general — remove analogies and examples, state the point directly",
    },
}

def make_prompt(text: str, axis: str, direction: str,
                current_score: float, target_score: float) -> str:
    desc = AXIS_DEFS[axis][direction]
    return f"""Rewrite this social media post so that it is {desc}.

Current {axis} score: {current_score:.2f}/1.0
Target  {axis} score: {target_score:.2f}/1.0  (shift of {abs(target_score-current_score):.2f})

RULES:
- Keep the EXACT same claim, facts, opinion and stance — do not add or remove information.
- Only change delivery on this one dimension.
- Keep roughly the same length (±20%).
- Return ONLY the rewritten post text, nothing else.

ORIGINAL POST:
{text}"""

def generate_rewrite(text, axis, direction, current_score, target_score, retries=3):
    for attempt in range(retries):
        try:
            r = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=400,
                messages=[{"role": "user", "content":
                           make_prompt(text, axis, direction, current_score, target_score)}]
            )
            return r.content[0].text.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None

# ── Load corpus ───────────────────────────────────────────────────
print("Loading corpus...")
rows = []
with open(POSTS_FILE) as f:
    for line in f:
        rows.append(json.loads(line))
df = pd.DataFrame(rows)
df["post_id"] = df["post_id"].astype(str)

def parse_axes(row):
    axes = json.loads(row) if isinstance(row, str) else row
    return {ax: axes[ax]["score"] if isinstance(axes.get(ax), dict) else axes.get(ax)
            for ax in AXIS_NAMES}

axes_df = df["axes_json"].apply(parse_axes).apply(pd.Series).astype(float)
df = pd.concat([df, axes_df], axis=1)
print(f"  {len(df)} posts loaded")

# ── Load existing cache ───────────────────────────────────────────
# Format: list of {post_id, original, rewrites: {axis: {direction, rewrite}}}
# Internal flat dict for fast lookup: (post_id, axis) -> entry
cache_list: list = []
cache: dict = {}   # (post_id, axis) -> rewrite str, for fast duplicate check

if Path(CACHE_FILE).exists():
    with open(CACHE_FILE) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        cache_list = raw
        for entry in cache_list:
            for ax, r in entry.get("rewrites", {}).items():
                cache[(entry["post_id"], ax)] = r["rewrite"]
    else:
        # Migrate old flat-dict format on the fly
        for key, r in raw.items():
            cache[(r["post_id"], r["axis"])] = r["rewrite"]

print(f"  {len(cache)} rewrites already cached")

# ── Select N_PER_TOPIC posts per topic ───────────────────────────
def mid_range_count(row):
    return sum(1 for ax in AXIS_NAMES if 0.2 <= row[ax] <= 0.8)

df["mid_count"] = df.apply(mid_range_count, axis=1)

selected_ids = []
rng = random.Random(SEED)
for topic, group in df.groupby("topic_name"):
    group_sorted = group.sort_values("mid_count", ascending=False)
    selected_ids += group_sorted["post_id"].iloc[:N_PER_TOPIC].tolist()

print(f"\nSelected {len(selected_ids)} posts ({N_PER_TOPIC} per topic × {df['topic_name'].nunique()} topics)")

# ── Build work list ───────────────────────────────────────────────
todo = []
for pid in selected_ids:
    row = df[df["post_id"] == pid].iloc[0]
    for ax in AXIS_NAMES:
        score     = float(row[ax])
        direction = "up" if score < 0.5 else "down"
        target    = min(score + 0.40, 1.0) if direction == "up" else max(score - 0.40, 0.0)
        if (pid, ax) not in cache:
            todo.append({
                "post_id": pid, "axis": ax,
                "direction": direction, "current": score, "target": target,
                "original": row["text"],
            })

total_needed = len(selected_ids) * len(AXIS_NAMES)
already_done = total_needed - len(todo)
print(f"Total pairs needed:  {total_needed}  ({len(selected_ids)} posts × {len(AXIS_NAMES)} axes)")
print(f"Already cached:      {already_done}")
print(f"To generate:         {len(todo)}")
print(f"Estimated cost:      ${len(todo) * 0.00062:.2f}")

# ── Generate ──────────────────────────────────────────────────────
def save_cache():
    # Rebuild list from flat cache dict
    by_post: dict = {}
    for (pid, ax), rewrite_text in cache.items():
        if pid not in by_post:
            # Find original text
            rows = df[df["post_id"] == pid]
            orig = rows.iloc[0]["text"] if not rows.empty else ""
            by_post[pid] = {"post_id": pid, "original": orig, "rewrites": {}}
        # Find direction
        rows = df[df["post_id"] == pid]
        if not rows.empty:
            score = float(rows.iloc[0][ax])
            direction = "up" if score < 0.5 else "down"
        else:
            direction = "up"
        by_post[pid]["rewrites"][ax] = {"direction": direction, "rewrite": rewrite_text}
    with open(CACHE_FILE, "w") as f:
        json.dump(list(by_post.values()), f, indent=2, ensure_ascii=False)

print(f"\nGenerating rewrites...")
failed = 0
for i, item in enumerate(todo):
    rewrite = generate_rewrite(
        item["original"], item["axis"], item["direction"],
        item["current"], item["target"]
    )
    if rewrite:
        cache[(item["post_id"], item["axis"])] = rewrite
    else:
        failed += 1

    if (i + 1) % 50 == 0:
        save_cache()
        print(f"  {i+1}/{len(todo)} done  |  {failed} failed  |  cached {len(cache)}")

save_cache()
print(f"\nDone. Total cached: {len(cache)}  |  Failed: {failed}")
print(f"Cache saved → {CACHE_FILE}")
