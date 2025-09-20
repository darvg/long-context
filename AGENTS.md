# Repository Guidelines

## Project Structure & Module Organization
Core library code lives under `src/topk_decoding`, where modules such as `topk_model.py`, `topk_attn.py`, and `topk_cache.py` implement sparse long-context mechanics. Keep new components co-located with related logic and expose them via `__init__.py` only when stable. Tests sit in `tests/test_topk.py`; mirror package layout when adding coverage. Benchmark scripts and datasets are under `benchmarks/`, with the RULER harness in `benchmarks/ruler/` and its downloaded assets stored in `benchmarks/ruler/data/`.

## Build, Test, and Development Commands
Install editable dependencies before hacking: `python -m pip install -r requirements.txt` followed by `python -m pip install -e .`. Run the full test suite from the repository root with `pytest`. Exercise the RULER benchmark using `bash benchmarks/ruler/run.sh llama-3-8b-1048k ivf 32768 niah_single_1 128 3`; swap in alternative models, indexes, or tasks as needed. Package builds are generated with `python -m build`, which uses the `pyproject.toml` metadata.

## Coding Style & Naming Conventions
Use Python 3.8+ features and keep auto-formatting consistent by running `black src tests`. Adopt four-space indentation, snake_case module and function names, CamelCase classes, and UPPER_SNAKE_CASE constants. Prefer explicit imports and add type hints for any new public API surface. When extending attention or cache logic, summarize the algorithm with a succinct module-level docstring.

## Testing Guidelines
Target pytest for all regression and unit tests; name files `test_*.py` and functions `test_<feature>_<behavior>`. GPU-heavy checks should guard on available hardware (e.g., `pytest.importorskip("torch.cuda")`) so the suite remains portable. For benchmarks, include small synthetic fixtures that exercise the same code paths as long-context runs. Aim to cover new branches in `topk_attn.py` and `topk_cache.py` with deterministic assertions before relying on benchmark metrics.

## Commit & Pull Request Guidelines
Adopt present-tense, imperative commit subjects (e.g., `feat: add IVF cache adaptor`). Group related changes together and avoid mixing refactors with functional updates. Each pull request should outline the motivation, the primary code paths touched, and verification steps (tests, benchmark commands, screenshots if CLI output matters). Reference tracked issues when available and tag reviewers who own the affected modules.

## Benchmark & Data Notes
Large benchmark downloads populate `benchmarks/ruler/data/`; avoid committing generated artifacts. Document any custom datasets or index parameters in the pull request, and stash reproducible command lines in `benchmarks/README.md` updates so future agents can replay results.
