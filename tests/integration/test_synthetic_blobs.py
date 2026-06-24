"""End-to-end pipeline test on synthetic blobs.

Placeholder — the full assertion (K-Means ARI > 0.95 against true blob labels)
is enabled once every Phase 1 stage is implemented. Kept here so the
integration test surface exists from the start.
"""
import pytest

pytestmark = pytest.mark.slow

def test_pipeline_recovers_three_blobs():
    pytest.skip("Full assertion enabled once all Phase 1 stages are implemented.")
