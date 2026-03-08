# ThriftLM Demo

Simulates 3 users sending 15 support queries through ThriftLM. Hit rate climbs from 0% to ~73% as repeated intents accumulate — Redis exact hits return in <1ms, semantic hits in ~1–3ms, misses call GPT-4o-mini.

## Setup

```bash
pip install thriftlm[api] colorama openai python-dotenv
```

Copy `.env.example` to `.env` and fill in your credentials:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
REDIS_URL=redis://localhost:6379
OPENAI_API_KEY=sk-...
```

Start Redis, then run:

```bash
# Rehearse without burning API credits
python demo/simulate.py --dry-run

# Real run (calls GPT-4o-mini for the 5 misses)
python demo/simulate.py
```

The real run takes ~20 seconds (15 queries × 0.8s pause + ~12s for the 5 LLM calls).
