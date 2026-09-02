"""
crpa - Contribution-gated Partitioned Attention.

Research question
-----------------
Attention *overlap* is a structural statistic: it describes how similar two
queries' attention supports look. *Redundancy* is a behavioral property: it
describes whether removing an interaction actually changes what the model does.
This package provides the machinery to measure both independently and to test
whether the first predicts the second.

Terminology (see README):
  structural overlap      - a geometric statistic over attention supports
  behavioral contribution - the measured effect of an intervention
  contribution-gated      - suppression gated on measured behavioral effect

The variant historically named ``crpa_causal`` is now ``crpa_contribution``.
The old name remains a resolving alias so existing checkpoints load unchanged.
"""

__version__ = "0.2.0"
