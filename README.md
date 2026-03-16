# Undergraduate Dissertation Experimental Codebase

Project: **Analysis and Research on Data Augmentation for Robust Visual Reinforcement Learning**

## Recommended Architecture

See `docs/architecture.md` for the full repository structure, file-level responsibilities, implementation order, and first file to implement.

## Immediate Build Principle

Build in this order:
1. Config schema and validation
2. Minimal pixel SAC baseline
3. Modular random shift augmentation
4. Robustness evaluation under spatial / temporal / physics shifts
5. Multi-seed aggregation and plotting

This keeps the codebase reproducible and extensible for later DrQ-v2 / TD-MPC2 comparisons.


## Robustness evaluation design

See `docs/robustness_evaluation_design.md` for wrapper-based domain-shift evaluation architecture and minimal initial corruption suite.
