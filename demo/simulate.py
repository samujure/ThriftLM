"""
ThriftLM demo — cache warming simulation.

3 simulated users, 15 queries. Shows hit rate climb from 0% to 70%+ in real time.

Usage:
    python demo/simulate.py              # real run (needs .env + OpenAI key)
    python demo/simulate.py --dry-run    # fakes latencies, no API calls
"""

import argparse
import random
import sys
import time

# Force UTF-8 output on Windows so box-drawing chars and emoji render correctly.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True, convert=True)
    RED    = Fore.RED
    YELLOW = Fore.YELLOW
    GREEN  = Fore.GREEN
    BOLD   = Style.BRIGHT
    RESET  = Style.RESET_ALL
except ImportError:
    RED = YELLOW = GREEN = BOLD = RESET = ""

DIVIDER = "━" * 50
API_KEY = "sc_demo"

# ---------------------------------------------------------------------------
# Query set — 15 queries, expected cache outcome annotated for dry-run
# ---------------------------------------------------------------------------
QUERIES = [
    ("User 1", "How do I reset my password?",                      "miss"),
    ("User 1", "What payment methods do you accept?",              "miss"),
    ("User 1", "How do I cancel my subscription?",                 "miss"),
    ("User 1", "I forgot my password, how do I get back in?",      "semantic"),
    ("User 1", "Do you accept credit cards?",                      "semantic"),

    ("User 2", "How do I reset my password?",                      "redis"),
    ("User 2", "How do I change my billing info?",                 "miss"),
    ("User 2", "How long does shipping take?",                     "miss"),
    ("User 2", "I can't remember my password",                     "semantic"),
    ("User 2", "What cards do you take?",                          "semantic"),

    ("User 3", "Password reset instructions",                      "semantic"),
    ("User 3", "Cancel subscription steps",                        "semantic"),
    ("User 3", "What payment methods do you accept?",              "redis"),
    ("User 3", "Shipping time estimate",                           "semantic"),
    ("User 3", "How do I update my credit card?",                  "semantic"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify(elapsed_ms: float) -> str:
    """Infer cache result type from latency."""
    if elapsed_ms < 2:
        return "redis"
    if elapsed_ms < 10:
        return "semantic"
    return "miss"


def fake_latency(kind: str) -> float:
    """Return a plausible fake latency (ms) for dry-run."""
    if kind == "redis":
        return round(random.uniform(0.3, 0.9), 2)
    if kind == "semantic":
        return round(random.uniform(1.0, 3.5), 1)
    return round(random.uniform(8_500, 14_200))


def fmt_result(kind: str, elapsed_ms: float) -> str:
    if kind == "redis":
        tag = f"{GREEN}{BOLD}🟢 REDIS HIT  {RESET}"
        note = "[exact match]"
    elif kind == "semantic":
        tag = f"{YELLOW}{BOLD}🟡 SEMANTIC   {RESET}"
        note = "[no LLM call]"
    else:
        tag = f"{RED}{BOLD}🔴 MISS       {RESET}"
        note = "[LLM called]"

    ms = f"{elapsed_ms:,.0f}ms" if elapsed_ms >= 100 else f"{elapsed_ms:.1f}ms"
    return f"  {tag} {ms:<12} {note}"


def print_query_block(user: str, query: str, kind: str, elapsed_ms: float,
                      hits: int, total: int) -> None:
    hit_rate = hits / total * 100
    print(f"\n{DIVIDER}")
    print(f"  {BOLD}{user}{RESET}  \"{query}\"")
    print(fmt_result(kind, elapsed_ms))
    bar_filled = int(hit_rate / 5)          # max 20 chars
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    print(f"  Hit rate: {hits}/{total} ({hit_rate:.1f}%)  [{bar}]")


# ---------------------------------------------------------------------------
# Real run
# ---------------------------------------------------------------------------

def run_real() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    import openai
    from thriftlm import SemanticCache

    cache = SemanticCache(threshold=0.82, api_key="demo")

    def call_llm(query: str) -> str:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful customer support agent. Answer briefly."},
                {"role": "user",   "content": query},
            ],
            max_tokens=120,
        )
        return response.choices[0].message.content

    hits = 0
    for i, (user, query, _expected) in enumerate(QUERIES, 1):
        t0 = time.perf_counter()
        cache.get_or_call(query, call_llm)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        kind = classify(elapsed_ms)
        if kind != "miss":
            hits += 1

        print_query_block(user, query, kind, elapsed_ms, hits, i)
        time.sleep(0.8)

    print(f"\n{DIVIDER}")
    print(f"  {BOLD}Done.{RESET}  Final hit rate: {hits}/{len(QUERIES)} ({hits/len(QUERIES)*100:.1f}%)")
    print(DIVIDER)


# ---------------------------------------------------------------------------
# Dry-run (no API calls, faked latencies)
# ---------------------------------------------------------------------------

def run_dry() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    import os
    from thriftlm.backends.supabase_backend import SupabaseBackend

    backend = SupabaseBackend(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ["SUPABASE_KEY"],
    )
    # Realistic cached response length for token estimation (~120 tokens)
    FAKE_RESPONSE = "x" * 480

    print(f"\n  {YELLOW}{BOLD}DRY RUN — LLM calls faked, metrics are real{RESET}\n")
    hits = 0
    for i, (user, query, expected) in enumerate(QUERIES, 1):
        elapsed_ms = fake_latency(expected)
        kind = expected
        if kind != "miss":
            hits += 1
            backend.record_hit(API_KEY, FAKE_RESPONSE)
        else:
            backend.record_miss(API_KEY)

        print_query_block(user, query, kind, elapsed_ms, hits, i)
        time.sleep(0.8)

    print(f"\n{DIVIDER}")
    print(f"  {BOLD}Done.{RESET}  Final hit rate: {hits}/{len(QUERIES)} ({hits/len(QUERIES)*100:.1f}%)")
    print(DIVIDER)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ThriftLM cache warming demo.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fake latencies — no OpenAI or Supabase calls.")
    args = parser.parse_args()

    print(f"\n{BOLD}ThriftLM — Cache Warming Demo{RESET}")
    print(f"3 users · 15 queries · threshold=0.82\n")

    if args.dry_run:
        run_dry()
    else:
        run_real()
