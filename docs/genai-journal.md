# GenAI Development Journal

This file is the source material for the 5-minute video's GenAI critical
evaluation segment (15% of the mark). Entries are dated; each captures a
specific interaction with the AI tool and what was learned, gained, or
corrected. The video should pull two or three of the most illustrative
entries and reflect on them.

> **Tool used**: Claude Code (Anthropic).

---

## How to use this file

For each notable interaction, add an entry in the format below. Aim
for a mix of:

- **Helpful**: where the AI accelerated something or surfaced an option you
  hadn't considered.
- **Hindering**: where the AI's first answer was wrong, misleading, or
  required correction.
- **Learning**: where wrestling with the AI's output taught you something
  about the underlying problem.

Be specific. "AI helped me write the crawler" is too generic. "AI suggested
`requests.get(url, timeout=10)` but I had to add retry-on-503 logic
myself" is what marks well.

---

## Entry template

```
### YYYY-MM-DD HH:MM — <one-line summary>

**Context**: what I was doing.
**Asked**: what I prompted the AI.
**Got**: what came back (paraphrase or quote a key snippet).
**Issue / outcome**: what was wrong, missing, or right; what I changed.
**Verdict**: helpful / partially helpful / hindering / neutral.
**What I learned**: the takeaway about the codebase or the problem.
```

---

## Entries

### 2026-05-06 — Initial planning from coursework brief

**Context**: Starting CW2 with two days to deadline.
**Asked**: AI to read the 19-page brief PDF and produce a development plan.
**Got**: A 13-section `PLAN.md` covering architecture, phases, mark
scheme analysis, risk register, timeline, and acceptance criteria.
**Issue / outcome**: The plan was thorough but had to be cross-checked
against the brief's mark scheme — the AI initially weighted phases evenly,
but the brief gives 20% to testing alone, so test depth was prioritised.
**Verdict**: Helpful. Saved several hours of structuring work.
**What I learned**: An AI-generated plan is a starting position, not a
final one. Useful only if you read it critically against the source brief.

### 2026-05-06 — Tokeniser apostrophe handling

**Context**: Designing the tokeniser used by both indexer and search.
**Asked**: AI for a regex to split text into tokens.
**Got**: A first cut using `re.findall(r'\w+', text.lower())`.
**Issue / outcome**: The simple approach treats apostrophes as
separators, splitting `don't` into `don` and `t`. The `t` fragment is
useless and `don` is not what users would search for. I changed the
implementation to **collapse** apostrophes (`don't → dont`) so
contractions remain searchable as a single unit. Also handled curly
Unicode apostrophes by normalising them first.
**Verdict**: Partially helpful. Got me to a working solution faster, but
the first answer would have produced bad search results.
**What I learned**: The tokeniser is the contract between the indexer
and the search module. A single rule there cascades into every query —
worth thinking about edge cases (apostrophes, Unicode, accents) early.

### 2026-05-06 — Test failure traced to URL normalisation

**Context**: Running the crawler test suite.
**Asked**: Built the crawler with `_normalise(url)` that strips trailing
slashes for stable comparison.
**Got**: Four tests failed. The root cause: my `_normalise` turned
`https://example.com/` into `https://example.com`, which then no longer
matched the FakeSession's keyed-by-`...com/` page lookups. The crawler
never fetched the home page; downstream tests failed empty.
**Issue / outcome**: Removed the `rstrip("/")` step. The URL we record
in the visited set must be the URL we actually issue HTTP requests
against, and a server can treat `/foo` and `/foo/` differently.
**Verdict**: Neutral — the bug was mine, not the AI's, but iterating on
the fix with the AI was faster than reading the stack traces alone.
**What I learned**: URL normalisation interacts with the visited-set key
*and* with what the HTTP client sends. Aggressive normalisation breaks
the symmetry.

### 2026-05-06 — TF-IDF formula for small corpora

**Context**: Implementing ranking in `src/search.py`.
**Asked**: AI for a standard TF-IDF score.
**Got**: `tf · log(N / df)`.
**Issue / outcome**: With ~70 documents and many terms appearing in
every page, `df == N` collapses `log(N/df)` to 0 — every score becomes
0 and ranking degenerates. Switched to a smoothed variant
`(1 + log10(tf)) · log10(1 + N/df)` so scores stay positive and
sub-linear TF dampens spam-like repetition.
**Verdict**: Partially helpful. The textbook formula is correct in
principle; the issue is small-corpus behaviour the AI did not flag.
**What I learned**: Information-retrieval formulas have implicit
assumptions about corpus size. Worth testing on a tiny synthetic
corpus before trusting the maths.

### 2026-05-06 — mypy strict pass

**Context**: Running `mypy --strict src/` for the first time.
**Asked**: AI to fix three reported errors.
**Got**: Three small fixes: install `types-requests`, coerce
`response.text` to `str(...)`, and handle bs4's
`AttributeValueList | str` union by going through `tag.get(...)` and
casting to `str`.
**Issue / outcome**: All three corrections were straightforward and
correct. mypy passes clean. Added `types-requests` to
`requirements-dev.txt` and a `mypy --strict` step to CI.
**Verdict**: Helpful.
**What I learned**: bs4's attribute access has a richer return type
than I expected — the `AttributeValueList` exists for repeated HTML
attributes (e.g. multi-value `class`). Worth knowing for future
scraping work.

---

## Engagement with the literature

Vaithilingam, Zhang & Glassman (2022), *Expectation vs. Experience:
Evaluating the Usability of Code Generation Tools Powered by Large
Language Models* (CHI EA), report a striking finding from their user
study of GitHub Copilot: developers **prefer** AI assistance and
*perceive* it as faster, but objective task-completion times do not
significantly improve. The bottleneck shifts: instead of writing
boilerplate, developers spend time **understanding, verifying, and
integrating** AI output. The authors call this the *verification gap*.

**My experience aligns with this finding.** The clearest example in
this project is the TF-IDF entry above (2026-05-06 — *TF-IDF formula
for small corpora*). When I asked for "a standard TF-IDF score" the
AI returned `tf · log(N/df)` — textbook-correct, indistinguishable
from any IR lecture. Without the gold-standard evaluation in
`evaluation/evaluate.py` I would have shipped it. The bug only
surfaces empirically when `df == N` (a term appearing in every
document collapses the score to zero, breaking ranking on small
corpora). Generation took seconds; *verifying* the formula was
appropriate for a 70-page corpus took roughly an hour of debugging
plus the work of building the evaluation harness in the first place.
Vaithilingam et al.'s framing fits: the AI didn't save me time on
the IR work — it shifted my time from *deriving* the formula to
*defending* the AI's choice. The total may even be higher, because
defending a borrowed formula is psychologically harder than owning
one I derived myself.

**Implications for *learning*.** Vaithilingam et al. don't address
education directly, but the verification-gap framing does. For a
coursework assessed at 30% of a module mark, the question is not
"did the AI produce passing code?" but "did the AI hide the
*reasoning* I was supposed to learn?". My answer, honestly: *partly*.
I understand TF-IDF's small-corpus pathology *because* I had to debug
the AI's textbook answer — that probably taught me more than reading
a chapter. But for the parts of the project the AI handled
fluently — atomic file writes, BeautifulSoup attribute coercion, the
politeness-window pattern — I have a working tool but a shallower
understanding than I would after building each from scratch. The
GenAI declaration in the video segment will be honest about this
asymmetry.

---

## To add before submission

- [x] Reflection on overall time savings vs. time spent verifying.
      *(Covered in the Vaithilingam discussion above.)*
- [x] One example where the AI's answer was *wrong* in a way that took
      time to debug. *(TF-IDF small-corpus collapse.)*
- [ ] One example where the AI surfaced a technique you had not heard of
      (or, conversely, where it missed an obvious modern technique).
- [x] A sentence on how using the AI affected your *learning* — did you
      end the project understanding the topic better, the same, or worse
      than if you had implemented from scratch?
      *(Implications-for-learning paragraph above.)*
