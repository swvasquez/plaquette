---
description: Style guide for the plaquette physics/algorithm docs under docs/.
paths:
  - docs/**
---

# Writing the plaquette docs

Files under `docs/` explain the *why* behind the code — the physics, algorithms,
and numerical contracts — so a reader understands why a thing must hold, not just
that it does. Follow the prose conventions in `CLAUDE.md` (connected paragraphs,
no bold-label headings, bullet lists only for genuinely enumerable things, plain
concrete language).

## Ground claims in the code

Read the source a claim rests on and confirm it before writing. Don't describe
intended or future behaviour as current; when the code and a claim disagree,
soften to what the code shows and tell the user.

## Labelled requirements

State load-bearing obligations as labelled requirements so code comments and tests
can cite them. Give each doc a short prefix and number sequentially in reading
order (e.g. `M1, M2, …`). Each requirement is one claim: a short bold lead clause,
then prose that justifies it and says what breaks if it fails. Let earlier ones be
cited by later ones, and keep a label stable once code depends on it.

## Math

Write math in LaTeX. Set long or load-bearing equations on their own line as
display blocks (`$$...$$`); use inline math (`$...$`) sparingly, since a paragraph
dense with it is hard to read. Prefer stating the identity or inequality that does
the work over describing it in words. Define symbols on first use and keep
notation consistent ($H$ action, $\beta$ inverse temperature, $N$ lattice size,
$\Delta E$ energy difference).

## Structure

Open with a short scope paragraph: what the file covers and how it relates to the
code. Develop the requirements in grouped sections, and close with a `## Status`
paragraph recording what the doc reflects as of writing. Keep top-level headings
to simple, concise phrases that name the theme (`## Algorithm`, `## Correctness`,
`## Status`) — not full sentences or generic labels.

## Tone

Explain, don't assert — the reasoning that joins the claims is the point. When a
requirement has several justifications, decide which one is load-bearing and
explain that one. Say plainly when something is sufficient but not necessary, or a
validation reference rather than ground truth.
