"""
qqp_benchmark.py — benchmark ThriftLM semantic cache against real QQP duplicate pairs.

Stores question1 from 200 QQP duplicate pairs, then benchmarks cache.lookup(question2)
across multiple similarity thresholds. Redis is flushed between threshold runs so
results are not contaminated by the Redis fast-path.

Run from repo root:
    py scratch/qqp_benchmark.py               # full run (store + benchmark)
    py scratch/qqp_benchmark.py --skip-store  # benchmark only (reuse existing Supabase entries)
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

for pkg in ("datasets", "tqdm"):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

# ---------------------------------------------------------------------------
# Presidio patch — en_core_web_lg (no C++ build tools needed)
# ---------------------------------------------------------------------------

from unittest.mock import patch

def _make_analyzer_lg(self=None):
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import SpacyNlpEngine
    nlp_engine = SpacyNlpEngine(
        models=[{"lang_code": "en", "model_name": "en_core_web_lg"}]
    )
    return AnalyzerEngine(nlp_engine=nlp_engine)

print("[Scrubber] Patching _get_analyzer to use en_core_web_lg.")
p_scrubber = patch(
    "thriftlm.privacy.PIIScrubber._get_analyzer",
    side_effect=_make_analyzer_lg,
)
p_scrubber.start()

# ---------------------------------------------------------------------------
# Imports (after patch is active)
# ---------------------------------------------------------------------------

import os
import numpy as np
from tqdm import tqdm

from thriftlm.cache import SemanticCache
from thriftlm.embedder import Embedder

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="QQP semantic cache benchmark")
parser.add_argument(
    "--skip-store",
    action="store_true",
    help="Skip the store phase and reuse existing sc_benchmark entries in Supabase.",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

print("\n[Data] Loading QQP validation split from HuggingFace...")
from datasets import load_dataset

ds = load_dataset("glue", "qqp", split="validation")
pairs = [(r["question1"], r["question2"]) for r in ds if r["label"] == 1][:200]
print(f"[Data] {len(pairs)} duplicate pairs loaded.")

# ---------------------------------------------------------------------------
# Supabase clean slate
# ---------------------------------------------------------------------------

from supabase import create_client

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

# ---------------------------------------------------------------------------
# Store phase
# ---------------------------------------------------------------------------

THRESHOLDS = [0.70, 0.75, 0.80, 0.82, 0.85, 0.90]
API_KEY = "sc_benchmark"

if args.skip_store:
    print("\n[Store] --skip-store set, reusing existing Supabase entries.")
else:
    print("\n[Supabase] Deleting existing sc_benchmark entries...")
    supabase.table("cache_entries").delete().eq("api_key", "sc_benchmark").execute()
    print("[Supabase] Clean slate ready.")
    print("\n[Store] Storing 200 question1 entries...")
    cache_store = SemanticCache(api_key=API_KEY, threshold=0.82)
    for q1, _ in tqdm(pairs, desc="Storing", unit="entry"):
        cache_store.store(q1, f"Answer to: {q1}")
    print("[Store] Done.")

# ---------------------------------------------------------------------------
# Pre-compute pairwise cosine similarities for diagnostics
# ---------------------------------------------------------------------------

print("\n[Similarity] Pre-computing pairwise cosine similarities...")
embedder = Embedder()

q1_texts = [p[0] for p in pairs]
q2_texts = [p[1] for p in pairs]

embedder._load()
embs1 = embedder._model.encode(q1_texts, normalize_embeddings=True, show_progress_bar=True)
embs2 = embedder._model.encode(q2_texts, normalize_embeddings=True, show_progress_bar=True)

# Cosine similarity = dot product for L2-normalised vectors.
sims = [float(np.dot(embs1[i], embs2[i])) for i in range(len(pairs))]
print(f"[Similarity] Done. Mean sim={np.mean(sims):.3f}, min={np.min(sims):.3f}, max={np.max(sims):.3f}")

# ---------------------------------------------------------------------------
# Benchmark phase
# ---------------------------------------------------------------------------

print("\n[Benchmark] Running lookup across thresholds (Redis flushed between runs)...\n")

results = {}  # threshold -> list of (hit: bool, q1, q2, sim)

# Use one cache instance just to get the Redis client for flushing.
_cache_for_redis = SemanticCache(api_key=API_KEY, threshold=0.82)
redis_client = _cache_for_redis._redis._get_client()

for t in THRESHOLDS:
    redis_client.flushdb()
    cache = SemanticCache(api_key=API_KEY, threshold=t)
    run_results = []
    for i, (q1, q2) in enumerate(tqdm(pairs, desc=f"t={t:.2f}", unit="pair", leave=False)):
        hit = cache.lookup(q2) is not None
        run_results.append((hit, q1, q2, sims[i]))
    results[t] = run_results

# Final Redis flush so we don't leave benchmark data in Redis.
redis_client.flushdb()

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

print("\n" + "=" * 52)
print(f"  {'Threshold':<10} {'Hits':<6} {'Total':<7} {'Hit Rate'}")
print(f"  {'-'*10} {'-'*6} {'-'*7} {'-'*8}")
for t in THRESHOLDS:
    run = results[t]
    hits = sum(1 for r in run if r[0])
    pct = hits / len(run) * 100
    print(f"  {t:<10.2f} {hits:<6} {len(run):<7} {pct:.1f}%")
print("=" * 52)

# Diagnostics at 0.82
DIAG_T = 0.82
diag = results[DIAG_T]
misses = [(sim, q1, q2) for hit, q1, q2, sim in diag if not hit]
hits_  = [(sim, q1, q2) for hit, q1, q2, sim in diag if hit]

misses.sort(key=lambda x: -x[0])   # closest misses first
hits_.sort(key=lambda x: x[0])     # lowest-sim hits first

def _fmt(q: str, n: int = 60) -> str:
    return (q[:n] + "...") if len(q) > n else q

print(f"\nTop 5 closest misses at threshold={DIAG_T}:")
for sim, q1, q2 in misses[:5]:
    print(f"  sim={sim:.3f}  Q1: \"{_fmt(q1)}\"")
    print(f"            Q2: \"{_fmt(q2)}\"")

print(f"\nTop 5 lowest-sim hits at threshold={DIAG_T}:")
for sim, q1, q2 in hits_[:5]:
    print(f"  sim={sim:.3f}  Q1: \"{_fmt(q1)}\"")
    print(f"            Q2: \"{_fmt(q2)}\"")

print(
    "\nNote: QQP pairs are general-domain Quora questions. Hit rates measure\n"
    "all-MiniLM-L6-v2 semantic sensitivity, not production cache behavior."
)

p_scrubber.stop()
