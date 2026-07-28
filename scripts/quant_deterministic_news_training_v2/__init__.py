"""Isolated deterministic-news v2 training utilities.

This package deliberately does not import the v1 experiment's training
contract.  It may reuse generic quant target, split, metric, and serialization
helpers, while all v2 paths, feature lists, protocol checks, status records,
and model artifacts remain in the v2 namespace.
"""
