# Contributing to veridist

veridist lives under [`python/`](python/) in this repository. All of the
guidance below is about that package. The repository root also still carries
the legacy `distfit_pro` project (the root `pyproject.toml`, `distfit_pro/`,
`tests/`, `examples/` and `docs/source` / `docs/user_guide` / `docs/api`); that package is legacy
and is **not accepting changes**. If you are not sure which tree you are in,
check whether your working directory is `python/` — if it is not, you are
probably looking at the legacy package.

## Repository layout

```
python/
  src/veridist/        # the package: domain, statistics, families, engine, execution, ...
  tests/                # contract, reference, unit, conformance, property, docs, quality, scale
  tools/                # coverage, release, and evidence-checking scripts used by CI
  quality/              # coverage-manifest.json and other gate manifests
  docs/                 # Sphinx source, EN/FA/DE locale catalogs, checkpoint-resume.md
  examples/             # runnable example scripts referenced from the docs
  KNOWN_LIMITS*.md       # the current release boundary, in en/fa/de
  README*.md             # the package README, in en/fa/de
docs/                   # repository-level ADRs, readiness ledger, evidence notes
```

The package changelog is [`python/CHANGELOG.md`](python/CHANGELOG.md).

Everything under `python/` targets Python 3.11+.

## Development setup

```bash
cd python
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[test,lint,docs]"
```

Use `.[mutation]` only if you intend to run `mutmut` on Linux (see
"Mutation testing" below), and `.[browser]` only for the opt-in Playwright
tests.

## Gates that must stay green

Run these from `python/` before opening a pull request:

```bash
python -m ruff check src tests docs tools
python -m mypy src
python -m pytest -q --cov=veridist --cov-branch --cov-report=json:coverage.json
python tools/check_coverage.py --project-root . --manifest quality/coverage-manifest.json --coverage-json coverage.json
```

`mypy` runs in `--strict` mode (`[tool.mypy]` in `pyproject.toml`); this
applies to every module under `src/`, including new ones.

Coverage thresholds are 95% global line/branch coverage, 98% for the
`domain`, `statistics`, `families`, and `engine` modules, and 90% per
production file, enforced by `tools/check_coverage.py` against
`quality/coverage-manifest.json`. That manifest pins the **exact** expected
`statements`/`branches` count for every production file. If you add,
remove, or restructure code in a file under `src/veridist/`, its recorded
denominators will no longer match the measured `coverage.json`, and the gate
fails with a "denominator drift" message. When that happens, regenerate
`coverage.json` from your own change and update only the denominators of the
files you touched — do not touch unrelated entries. Do not add
`# pragma: no cover` or `# pragma: no branch` to work around the gate, and
never add `# pragma: no mutate`, which the mutation tooling rejects.

Never weaken or delete an existing test to make a gate pass. If a test
encoded behavior that a fix corrects, change its expectation and explain why
in the commit message.

### Mutation testing

`mutmut` cannot run on Windows in this project (`tools/run_mutation.py`
refuses to start). Mutation testing (`.github/workflows/mutation.yml`) runs
only on Linux CI, against `ubuntu-latest`. When you add a regression test for
a statistical or execution-path fix, put it under `tests/contract`,
`tests/reference`, `tests/unit`, `tests/conformance`, or `tests/property` so
that job can see it; tests under other directories (for example `tests/docs`
or `tests/quality`) are excluded from the mutation run by design.

## Documentation parity (en/fa/de)

User-facing documentation exists in English, Persian (fa), and German (de):
`README*.md` at the repository root and in `python/`,
`python/KNOWN_LIMITS*.md`, `docs/capability-guide*.md`, the Sphinx sources
under `python/docs/source/*.md`, and the matching catalogs under
`python/docs/locales/{fa,de}/LC_MESSAGES/*.po`. Tests under `python/tests/docs`
and `python/tests/quality` enforce this parity — in places, they check for
exact strings or exact gettext message sets extracted from the English
source.

When you change an English sentence that is covered by this parity
requirement, make the equivalent change to the Persian and German text in
the same commit. Keep Persian and German natural rather than literal,
word-for-word translations, but keep API names, literals, error codes, and
numbers identical across all three languages. If you touch
`python/docs/source/api.md` (or the other Sphinx source pages), check
`python/docs/i18n/parity-manifest.json` and run the docs test suite — it
regenerates the real `.pot` catalogs with Sphinx and compares them against
the manifest and the `.po` files, so a mismatch is caught locally:

```bash
pip install -e ".[docs]"
python -m pytest tests/docs tests/quality -q
```

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/): a prefix
such as `fix:`, `feat:`, `docs:`, `test:`, `refactor:`, `perf:`, or `ci:`,
optionally scoped (for example `fix(families): ...`, `docs(readme): ...`),
an imperative-mood subject of 72 characters or fewer, and a body that
explains the reasoning behind the change, not just what changed.

## Pull requests

- Keep a pull request scoped to one change; unrelated cleanups belong in a
  separate PR.
- Add tests for the behavior you changed, and update
  `python/CHANGELOG.md` under `## [Unreleased]` for any behavior change.
- Do not bump the package version; that is a maintainer decision made at
  release time.
- Run the gates above locally before requesting review. CI re-runs them on
  Linux across the supported Python versions, plus the Linux-only mutation
  job for the statistical and execution core.

## License of contributions

veridist is distributed under the Business Source License 1.1, which is
source-available and not an open-source license; its additional use grant
covers non-commercial production use only (see `LICENSE`). By submitting a
contribution you agree that it is provided under the same license.

## Reporting issues

Include a minimal reproducible example, the Python version, operating
system, and `veridist` version, and the exact error or unexpected output.
