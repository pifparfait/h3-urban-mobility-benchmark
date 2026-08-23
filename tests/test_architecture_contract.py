from h3uab.config import Config

def test_architecture_contract():
    cfg = Config()
    assert cfg.h3.interaction_res == 11
    assert cfg.h3.position_res == 12
    assert cfg.h3.accessibility_res == 9
    assert cfg.h3.amenity_index_res == 10
    assert cfg.routing.routing_is_common_physical_infrastructure is True
    assert cfg.routing.enabled is False
