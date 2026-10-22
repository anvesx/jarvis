# LLD template

Every component of orb has an `LLD.md` next to its code. Copy this file, keep the eight sections in this
order, and fill each one. A section that does not apply says so in one sentence and says why.

Rules for every LLD:
- **Measured, not guessed.** Every number has a source: measured (with the machine and the date), chosen (with
  the reason), or taken from a paper or a model card (with the link).
- **One claim per sentence.** Bold the claim, then give the detail.
- **Tests are named.** Each test case points to the test function that checks it.

---

# LLD: `<component name>`

Code: `<path>` · Tests: `<path>` · Status: `<draft | built | in production>` · Last update: `<YYYY-MM-DD>`

## 1. Input
What comes in, from where, in what format, and with what limits. Say how much each input is trusted
(user text, another service, a file on disk).

## 2. Algorithm
The steps in order. Name the state that each step reads and writes. Say what runs in parallel and what
waits for what.

## 3. Maths
Every formula, threshold, limit and constant, with its value and its source.

## 4. Test cases
| Case | Input | Expected result | Test |
|---|---|---|---|

## 5. Security
Trust boundaries. Secrets and where they live. Data that leaves the machine. How an attacker could misuse
the component, and what stops them. What is not protected yet.

## 6. Performance
Measured numbers (machine, date, input size). The budget. The bottleneck. What to try next, with the expected
gain.

## 7. Edge cases
| Case | What happens | Why |
|---|---|---|

## 8. Expected results
What a user or an operator sees when the component works. The checks that accept it.
