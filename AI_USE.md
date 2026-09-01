# Declaration of AI use

AI was used in building this repository and in drafting parts of the accompanying paper.
This file states where, how the output was checked, and what it was not used for.

Every claim below is anchored to something in the repository — a file, a commit, a test
count — so that it can be checked rather than taken on trust.

## Summary

Anthropic's Claude (models Opus 5 and Sonnet 5, used through Claude Code) was used as an
implementation and drafting tool: writing code against specifications I wrote, writing
documentation, debugging and refactoring existing code, and drafting the results and
discussion prose.

The research question, the design of the proposed residual-income model, the choice of
benchmarks and evaluation methodology, and the interpretation of the results are mine. I
have reviewed everything in this repository and in the paper, and I am responsible for all
of it, including any errors. AI is not an author of this work.

## What AI was used for

**Implementing code from specifications I wrote.** Each model and pipeline was specified
before it was implemented, in a working directory of specification documents
(`Instruction/`, about 128,000 characters across ten files, kept out of version control
as a personal working record). Those documents drove the implementation:

| Specification | Implementation |
|---|---|
| `epo_equity4_implementation.md`, `epo-implementation-guide.md` | `src/EPO/` |
| `PPP_2009.md`, `PPP_note.md` | `src/PPP/` |
| `data-collection-rim.md` — "Master Data Collection Spec" | `src/Data/` |
| `CRRA_CE_Inference_Obsidian.md`, `mean_return_difference_HAC_test_obsidian.md` | `src/Empirical_Analysis/` |

**Documentation.** Docstrings, inline comments, and the per-package `README.md` files.
This is the largest AI-assisted surface in the repository by volume, and the explanatory
style throughout the code is largely AI-drafted from my notes and then edited.

**Debugging, review and refactoring.** Two examples that are visible in the history:
commit `7cbb3f1` replaced the EPO risk model with the paper's Equity specification, sized
the covariance window to the universe, and corrected `src/EPO/README.md`, which had the
robust-optimisation framing backwards. A later reproducibility pass moved the SEC EDGAR
`User-Agent` out of the source into a caller-supplied value, fixed the packaging so
`pip install -e .` produces a working install, and corrected several inaccurate statements
in `README.md`.

**Paper prose — results and discussion.** AI was used to draft the results and discussion
narrative. It worked to a written instruction of mine
(`Instruction/results_discussion_writing_instruction.md`) that requires facts to be
separated from interpretation in a findings table before any prose is written, and that
states the goal explicitly:

> The goal is **not** to hide the insignificant inference results or to claim that the
> proposed models definitively dominate the benchmark.

The empirical finding that drafting had to convey — strong realised performance that
conventional inference does not distinguish from the equal-weight benchmark — was fixed by
the results before any text was written, and was not something the drafting was free to
soften.

## What AI was not used for

- **The research question**, and the design of the proposed residual-income model.
- **Methodology**: the choice of benchmarks (EPO, PPP, equal-weight), the universe
  construction and exclusion rule, the rebalance calendar, and the evaluation and
  inference procedures.
- **Interpretation**: what the results mean, and the conclusions drawn from them.
- **Data.** No data in this project was generated, synthesised, imputed or estimated by
  AI. Every figure traces to one of four external sources — SEC EDGAR, Yahoo Finance,
  Kenneth French's data library, and the `hanshof/sp500_constituents` history — and the
  entire collection is reproducible from scratch with `python -m src.Data.run`. Where the
  pipeline fills gaps it does so by bounded forward-fill from filed values, in code, with
  the limits set in `constant.py`.
- **Authorship.** AI is not an author and is not credited as one.

## How the output was verified

**Test suite.** 268 tests, passing:

| Tests | File | Covers |
|---|---|---|
| 109 | `tests/test_sampling_distribution.py` | the simulated return distribution |
| 55 | `tests/test_proposed_model_utils.py` | valuation and optimisation utilities |
| 32 | `tests/test_empirical_analysis.py` | backtest engine and metrics |
| 25 | `tests/test_sharpe_inference.py` | Sharpe-ratio inference |
| 22 | `tests/test_mean_inference.py` | mean-difference inference |
| 19 | `tests/test_ce_inference.py` | certainty-equivalent inference |
| 6 | `tests/test_rebuild.py` | rebuilding metrics from saved weights |

**Review.** I read every material change before it entered a commit.

**Replication against the source papers.** The implementations were checked against the
published specifications rather than against their descriptions: Pedersen, Babu & Levine
(2021), Table 1, for the EPO Equity 4 specification; Brandt, Santa-Clara & Valkanov (2009)
for the Parametric Portfolio Policy; Ledoit & Wolf (2008) for the studentized bootstrap
used in the Sharpe-ratio test.

**Simulation.** The inference machinery is checked against simulated data, not only along
its code paths. Those checks are marked `slow` in `tests/test_sharpe_inference.py`,
`tests/test_ce_inference.py` and `tests/test_mean_inference.py`; `pytest -m "not slow"`
skips them.

## Attribution in the commit history

Two commits carry a `Co-Authored-By` trailer:

```
7cbb3f1  EPO: equity risk model, universe-scaled covariance window, MIT license
         Co-Authored-By: Claude Opus 5
8e91d4c  Untrack .vscode and ignore Python/editor cache files
         Co-Authored-By: Claude Sonnet 5
```

**These two are not a complete index of AI involvement.** AI assistance was considerably
broader than the trailers indicate, as described above; the trailers were added only from
the point at which the tooling emitted them. This file, not the commit history, is the
accurate record.

## For the paper

A statement suitable for a submission's disclosure section:

> During the preparation of this work the author used Anthropic's Claude (models Opus 5
> and Sonnet 5, via Claude Code) to implement the analysis code from specifications
> written by the author, to write code documentation, to debug and refactor that code, and
> to draft portions of the results and discussion text. The research question, model
> design, methodology and interpretation of results are the author's own. No data was
> generated or synthesised by AI. The author reviewed and edited all output, takes full
> responsibility for the content of the publication, and confirms that AI is not an author
> of this work.

Journals word this requirement differently and some prescribe an exact form, placement or
section heading. Treat the paragraph above as a starting point and check the policy of the
target venue before submitting.
