"""
experiments - runnable studies, one module per question.

Every module writes structured results under ``results/`` and never plots
directly, so figures regenerate from files without retraining
(``python -m experiments.plot_all``).
"""
