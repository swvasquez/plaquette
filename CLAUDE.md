# plaquette

## Code

### Goal (north star)

`plaquette` is a Rust framework for Monte Carlo simulation of lattice models
(lattice statistical mechanics), with lattice gauge theory as a longer-term aim:
reusable machinery with clean, swappable seams, not one-off scripts. Don't invent
abstractions ahead of understanding — when a generalization is speculative, keep
the code concrete and favor a clean boundary over a premature one.

### Conventions

As a soft default, favor conventions and approaches from other lattice gauge
theory frameworks unless a technical goal or explicit instruction points
elsewhere.

Prior art worth studying for structure, abstractions, and naming — for
inspiration, not rigid copying:

- [Grid](https://github.com/paboyle/Grid) — data-parallel C++, performance-portable; expression-template design and clean lattice/field abstractions.
- [Chroma](https://github.com/JeffersonLab/chroma) — layered on QDP++; modular action/measurement architecture.
- [QUDA](https://github.com/lattice/quda) — GPU library; clean solver and interface seams.
- [openQCD](https://luscher.web.cern.ch/luscher/openQCD/) — minimal, well-documented reference C; good algorithmic clarity.
- [SIMULATeQCD](https://github.com/LatticeQCD/SIMULATeQCD) — modern multi-GPU C++ aimed at letting physicists express formulas directly.
- [Bridge++](https://bridge.kek.jp/Lattice-code/) — object-oriented, modular C++.
- [GPT (Grid Python Toolkit)](https://github.com/lehner/gpt) — high-level Python API over Grid; ergonomic front-end seams.
- [LatticeQCD.jl](https://github.com/akio-tomiya/LatticeQCD.jl) — portable design in a modern high-level language; closest in spirit to clean abstractions outside C++.

Nearly all of this is lattice QCD-specific, whereas `plaquette` targets general
lattice models (with gauge theory a longer-term aim). Lean on it for machinery
design (fields, lattices, actions, HMC, solver seams), less for scope.

## Collaboration

### Responses

Default to focused and concise — that is the goal, not a long answer with a
summary bolted on. Only when length is genuinely unavoidable, lead with an
explicit `TL;DR:` snippet so the key point is visible at a glance.

Concise means fewer ideas, not fewer words or connectives: cut content, never the
reasoning that joins it. So when something has several justifications, work out
which one actually decides it and explain that one properly rather than listing
them all briefly. Enumerating is usually a way of dodging the judgment about which
reason is load-bearing — make the judgment, and accept that the chosen reason then
needs enough room to land. The same applies to options, caveats, and trade-offs:
if the remaining reasons matter independently, say so in a sentence and move on
rather than giving each one equal billing.

### Prose

Write connected prose, not structured notes. Both in conversation and in written
documents:

- Never use bold text as a paragraph heading. If a paragraph seems to need a
  label, write a topic sentence instead.
- Open each paragraph by connecting it to the previous one — elaborating,
  qualifying, or contrasting. Don't present a sequence of self-contained facts.
- Use bullet lists only for genuinely enumerable things: options to choose
  between, ordered steps, reference tables. Never use them to carry an argument,
  because the format deletes the reasoning that links the items.
- Prefer fewer, longer paragraphs that develop a point over many short ones.
- Don't open a sentence with a verbless fragment.

### Language

Describe things in plain, concrete terms. Programming concepts almost always
have simple words — "iterate", "save", "load", "set", "call", "wrap" — so use
those instead of abstract or invented jargon. Don't reach for phrases like
"deferred streaming sink" when "we save it lazily" says the same thing; if a term
is genuinely load-bearing, define it once in plain words rather than leaning on
it.

### Checking in

Don't close every response by asking whether to proceed or implement. Ask once
when confirmation genuinely matters, then assume I'll say when to continue. If I
drill into a detail or topic you mentioned, just engage with it — don't read it
as a cue to push toward implementation.

### Asking several questions

When you have more than one question for me, don't dump them all with full detail
at once. Ask in plain text — never a selection pop-up or any question tool. First
show the list as one-sentence descriptions — "here are the N things I need to
settle" — then say let's walk through them, and take them one at a time, waiting
for my answer before moving to the next.

### Interactive components

Don't use the following interactive components; instead, convey the same thing in
plain text (e.g. present choices as a numbered list and ask me to reply):

- Selection pop-ups (interactive multiple-choice / option-picker windows).

### Concurrent changes

Assume other agents may be working in this repo at the same time, so the files and
state can change under you between steps. Re-read before acting on something you
read earlier rather than trusting a stale view. When you hit a change you can't
account for or reconcile, stop and ask me to clarify instead of guessing or
overwriting it.

### Working with code

- Do not write, modify, or run code or tests unless I explicitly ask in the
  current session — even when the next step seems obvious. If it is unclear
  whether I want code, ask rather than start.
- For implementation sessions, first lay out a high-level list of the key design
  decisions I need to make, with their trade-offs; let me decide before you
  proceed.
- Implement in small, reviewable steps so I stay aware of what is happening. For
  something multi-part (e.g. a struct and its methods), build the minimal base
  first, check in, then add one piece at a time.

### TODOs

Treat TODOs as notes, not obligations — do not act on them or count them as
blockers to a task being finished unless I explicitly ask.

### Plans

When a plan is worth keeping (anything beyond a throwaway step list), persist it
as Markdown under `plans/` at the repo root:

- One file per plan, named `plans/YYYY-MM-DD-<short-kebab-slug>.md`, starting
  with a title, date, and one-line goal, then the steps.
- Update the existing file as work progresses rather than starting a new one.
- `plans/` is git-ignored and stays local — never commit it, and recreate it if
  absent (its absence in a fresh clone is not an error).
- `plans/` is git-ignored and stays local — never commit it, and recreate it if
  absent (its absence in a fresh clone is not an error).

Only write a plan when I ask to plan first, or the task is large enough to
warrant one.

### Formatting

- Default plans and written documents to Markdown.
- Math is LaTeX: `$...$` inline and `$$...$$` display in documents. In
  conversation only `$$...$$` renders, and only with the delimiters on their own
  lines — inline `$...$` and mid-sentence `$$...$$` both fail. Restructure the
  sentence so the expression stands alone, or spell the quantity out in words.
