# Experimentation Framework (A/B toolkit with CUPED, guardrails, sequential examples)

This repository provides a production-ready scaffold to design, analyze, and monitor online experiments.

## Highlights:
* Power/MDE calculators (means and proportions)
* Guardrails: SRM detection, invariant metric checks
* CUPED variance reduction for mean outcomes
* A/B test modules for proportions and means
* Sequential testing examples (Pocock-like boundaries)
* dbt models for assignment/outcome standardization and an experiment facts table
* Notebooks for power analysis, A/B analysis, and sequential monitoring
* Mock dashboards (PNGs) and guidance for BI integration

## Quickstart
* Create a virtual environment and install requirements from env/requirements.txt.
* Seed and run dbt in dbt_project/ (pick your adapter; replace seeds with warehouse sources later).
* Use notebooks for power/MDE, A/B analysis with CUPED, and sequential examples.
