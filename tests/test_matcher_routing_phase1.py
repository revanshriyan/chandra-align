import pytest


def test_matcher_factory_defaults_to_lightglue_and_sift_fallback():
    from chandra_align.matcher import LightGlueMatcher, SIFTMatcher, make_matcher

    cfg = {"tier1": "lightglue_aliked", "tier2_escalation": {"matcher": "sift"}}
    assert isinstance(make_matcher(cfg, "tier1"), LightGlueMatcher)
    assert isinstance(make_matcher(cfg, "tier2"), SIFTMatcher)


def test_rift2_factory_requires_explicit_opt_in(monkeypatch):
    import chandra_align.matcher as matcher_module

    class SentinelRift:
        pass

    monkeypatch.setattr(matcher_module, "RIFT2Matcher", SentinelRift)
    with pytest.raises(ValueError, match="rift2_opt_in"):
        matcher_module.make_matcher({"tier1": "rift2"}, "tier1")
    assert isinstance(
        matcher_module.make_matcher({"tier1": "rift2", "rift2_opt_in": True}, "tier1"),
        SentinelRift,
    )
