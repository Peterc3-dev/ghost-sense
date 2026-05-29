# ghost-sense

An experimental, heuristic engine that infers a coarse "ambient state" of a chat
user from their messages, decays that estimate over time, and emits hidden
context that a downstream LLM can use to adapt its replies.

## Status

Early / experimental (v0.1.0). It runs and has a test suite (13 test modules
under `tests/`), but it is a prototype, not a finished product:

- All inference is **heuristic** — regex keyword matching, simple ratios, and
  linear regression. There is no machine learning and no calibration against
  real labelled data, so the extracted "state" should be treated as a rough
  guess, not a measurement.
- There is no CLI, daemon, or service wrapper. It is a library you import and
  drive from your own code.
- It was written as one component of a larger personal orchestration setup, so
  parts of it (the routing and adapter modules) reference external systems by
  name rather than shipping a working integration.

## What it does

The pipeline (`ghost_sense/main.py`, class `GhostSense`) processes one message
at a time:

1. **Extractors** (`ghost_sense/extractor/`) scan the message text and metadata
   and emit `SignalEvent`s. Each extractor targets one axis:
   - `register` — linguistic style: slang ratio, average word length,
     punctuation/capitalization, emoji density, and a composite formality score.
   - `sleep` — keyword/pattern matches for poor vs. good sleep, caffeine
     mentions (weighted higher late at night), and fatigue vs. energy language.
   - `nutrition`, `stress`, `cadence` — analogous keyword/timing heuristics.
   - `absence` — treats *missing* signals as data. It calibrates per-dimension
     observation intervals over a warm-up window (default 14 days) and then
     flags a dimension as "overdue" when it hasn't been seen for a multiple of
     its baseline interval.
2. **Event bus** (`ghost_sense/bus.py`) — a minimal in-process pub/sub that
   routes events to the accumulator.
3. **Accumulator** (`ghost_sense/accumulator/`) — persists events to SQLite,
   merges each new observation into a per-dimension `StateVector`, and applies
   **continuous exponential decay** (`confidence * e^(-lambda * hours)`) at read
   time so older estimates fade. Decay rates per dimension live in
   `ghost_sense/config.py`. It also computes **trends** (rising/falling/stable)
   via ordinary-least-squares regression over snapshot history (3- and 7-day
   windows by default), cached in SQLite.
4. **Conditioner** (`ghost_sense/conditioner/`) — turns the current state into a
   hidden system-prompt preamble of adaptation directives (e.g. "keep responses
   concise", "match casual register"). Adapters format this for specific LLM
   APIs (OpenAI-compatible message shape).
5. **Harvester** (`ghost_sense/harvester/prompts.py`) — a bank of conversational
   probe questions tied to low-confidence dimensions, with per-probe cooldowns,
   that the conditioner *may* suggest injecting to gather more signal.
6. **Routing** (`ghost_sense/routing/cin_feedback.py`) — maps the state vector
   to delegation/probing recommendations for an external orchestrator.

State, events, snapshots, and the trend cache are stored in a local SQLite
database (schema in `schema.sql`, WAL mode).

## Note on intent

This is, by design, covert: it estimates a user's condition from their writing
without telling them, prepends instructions to the model that are explicitly
marked "do not surface to user," and proposes probe questions to elicit more
data. That design is the point of the experiment, but it carries obvious
privacy and consent implications. Only run it on your own conversations, or with
the clear, informed consent of anyone whose messages it sees.

## Install / use

Pure Python, standard library only (no third-party runtime dependencies);
requires Python 3.11+.

```bash
pip install -e .
```

```python
from ghost_sense.main import GhostSense

engine = GhostSense(db_path="ghost_sense.db")
state = engine.process_message("barely slept, on my third coffee already")

for dim, fs in state.fields.items():
    print(dim, round(fs.value, 2), "conf", round(fs.confidence, 2))

engine.close()
```

To produce the hidden preamble / routing hint from a state vector, see
`ghost_sense/conditioner/preamble.py` and
`ghost_sense/routing/cin_feedback.py`.

## Tests

The tests use `pytest` (not currently declared as a dependency — install it
separately):

```bash
pip install pytest
pytest
```

## Limitations

- English-only, keyword/regex driven — easily fooled, no semantic understanding.
- Thresholds and decay rates in `config.py` are hand-picked, not tuned.
- The absence detector needs a multi-day calibration window before it does
  anything.
- The routing and Boo2 adapter modules target external systems that are not
  included here.

## License

MIT — see [LICENSE](LICENSE).
