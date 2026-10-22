# 0004 · The debt engine replaces the weighted steering

Date: 2026-10-21 · Status: accepted

## Decision
**The discussion agent decides with debt rules. The weighted formula (`need = 0.5 × gap + 1.0 × pressure`,
with hand-set targets) is deleted.**

A message that leaves something open starts a debt: a question owes an answer, a claim owes a reason, a
disagreement owes an end. A later message pays it. The stage (Open, Test, Close) sets which debts are due.
The most urgent due debt is the need, and its kind picks the persona. Jev only answers yes/no questions.
The only number in the rules is 0.5, where a Jev answer counts as "yes".

## Why
- The weighted version could not be explained: in a replay of 70 saved person messages, 23 persona choices
  were won by less than 0.10, and a Jev answer moves by up to 0.05 between identical calls.
- Every debt decision points to one Jev answer and one rule, so it can be explained and tested.

## How it was checked
- On the passwords chat, the debt rules gave the same 7 personas as the old app (recorded 2026-10-19).
- `test_replay_recorded.py` replays those recorded Jev answers and checks every question, debt, stage,
  need and check, with no network call.

## What was removed
`app/steer.py`, `app/policy.py`, `app/explain.py`, the old chat UI and `pipeline/*` (the graph engine and the
GitHub-issues tooling). They are in the archive named in 0001.
