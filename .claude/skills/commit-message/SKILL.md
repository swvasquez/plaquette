---
name: commit-message
description: Write a git commit message for the current session's work, following the plaquette commit standard. Use when the user asks for a commit message — e.g. "give me a commit message for the latest session", "write a commit message for these changes", "what should I commit this as". The skill produces the message text for the user to commit themselves; it does not run git commit.
---

# Writing a plaquette commit message

The user runs this when they want a commit message for the work done in the
current session. Produce the message text and present it for them to use — the
user is the committer, so do not run `git commit` or stage anything yourself
unless they explicitly ask.

Base the message on the work done in the current session — what was changed and
why, drawn from the session's context. Confirm it against `git status` and
`git diff` (and `git diff --staged` if anything is staged) so the message
matches what actually landed rather than intentions discussed earlier that may
not have made it in.

If no session context is available — for instance the skill is invoked cold in a
fresh session — fall back to the recent `git log` history to infer what the work
was. And if the commit history is empty too, don't guess: tell the user there's
nothing to go on and ask them to point you at the relevant work.

Write a single imperative subject line: "Add solver seam", not "Added" or
"Adds", capitalized, no trailing period, about 50 characters. It should complete
"this commit will …", naming the change rather than describing it after the
fact. Do not add a Conventional Commits type prefix or any attribution /
co-authored-by trailer.

Most commits are subject-only. Add a body only when the change carries a *why*
the diff doesn't show — a non-obvious reason, a tradeoff weighed and rejected,
the root cause behind a subtle fix, or a consequence a future reader should
know. When the subject alone answers "what changed and why," a body is noise;
when someone reading `git log` in a year would be left guessing, it earns its
place. Judge each commit on that test rather than defaulting either way. If the
user asks explicitly for a body or says what it should contain, follow that
over the subject-only default.

When you do include a body, separate it from the subject with exactly one blank
line — git relies on that blank line to split the two — and wrap it at about 72
characters, using it to explain the reasoning rather than restate the diff.

Present the finished message in a fenced code block so the user can copy it
cleanly, and keep any surrounding commentary short.
