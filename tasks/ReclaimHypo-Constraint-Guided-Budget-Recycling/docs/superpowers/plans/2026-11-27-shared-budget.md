# Shared Budget Implementation Plan

> Execute natively using superpowers:executing-plans; tests follow test-driven-development.

**Goal:** Verify no-repeat generation and shared output budgets without changing frozen fixed-budget results.
**Architecture:** Persistent canonical streams from frozen task adapters; round robin scheduler; independent official validators and posterior answer-set scoring.
**Tech Stack:** Python3.9, unittest, frozen numpy/sympy/networkx/Z3 CPU requirements.
**Spec:** docs/superpowers/specs/2026-11-27-shared-budget-design.md

## Global Constraints

- No GPU, training, gold/pool/S in scheduler inputs, or writes to previous release.
- Policies fixed_repeat/fixed_stop/shared_round_robin; cap6, total6×N, timeout30seconds.
- Preserve all outputs, denominator definitions, missing quota and solver cost.

## Review Focus

- Duplicate question IDs fail; identical answers for distinct IDs are distinct items.
- Empty solution spaces must not fake perfect rates or consume output slots.
- Timeouts must not count as exhaustion or allow invented repeats.
- Exhaustion exactly at last output cannot be claimed without another generator step.
- Shared budget changes allocation; old fixed-quota metrics cannot be relabeled.

### Task 1: Scheduler

Files: budget_v1/scheduler.py, test_scheduler.py.
Interface: allocate(cases, policy, per_case=6, total_budget=None, time_limit=30) -> dict.

- [x] Write hand-checked finite-space, fairness, timeout, empty-space and leak tests.
- [x] Watch tests fail before implementing persistent unique streams.
- [x] Run scheduler and frozen backend suites; expect all pass.

### Task 2: Evaluation and delivery

Files: budget_v1/evaluate.py, README.md, report.md, reproduce_cpu.sh.
Consumes: scheduler result; separate frozen validators and answer sets.

- [x] Score three policies under original order and five frozen permutations on all40 cases.
- [x] Assert same total output108 for internal fixed_repeat/shared, no shared repeats.
- [x] Match fixed results with previous frozen release; validate all outputs independently.
- [x] Run standalone reproduction; review implementation and claims; package hashes.
