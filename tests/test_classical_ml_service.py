"""Pattern #1 — Client → Feature Pipeline → Model → Registry."""

from resumatch.features.extractor import FEATURE_NAMES, FeatureExtractor
from resumatch.model.predictor import Predictor
from resumatch.model.registry import ModelRegistry


def test_extractor_fills_the_serving_schema():
    extractor = FeatureExtractor()
    vector = extractor.extract({})
    assert list(vector) == list(FEATURE_NAMES)
    assert extractor.as_row({FEATURE_NAMES[0]: 2})[0] == 2.0


def test_registry_versions_and_promotes():
    registry = ModelRegistry()
    first = registry.register("match_ranker", metrics={"auc": 0.8}, artifact={"weights": {}})
    second = registry.register("match_ranker", metrics={"auc": 0.9})
    assert first.version == 1 and second.version == 2
    prod = registry.promote("match_ranker", 2, "Production")
    assert registry.latest("match_ranker", stage="Production").version == prod.version


def test_predictor_uses_registered_weights():
    registry = ModelRegistry()
    weights = dict.fromkeys(FEATURE_NAMES, 0.0)
    weights[FEATURE_NAMES[0]] = 4.0
    registry.register("match_ranker", artifact={"weights": weights, "intercept": 0.0})
    predictor = Predictor(registry)
    high = predictor.predict({FEATURE_NAMES[0]: 5.0})
    low = predictor.predict({FEATURE_NAMES[0]: -5.0})
    assert high["score"] > low["score"]
    assert high["target"] == "interview"
