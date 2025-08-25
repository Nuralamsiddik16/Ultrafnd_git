from .advanced_fusion import HierarchicalFusionModel, SimpleFusionModel


def get_model(config):
    """Factory for creating models based on configuration."""
    if getattr(config, "model_name", "hierarchical_fusion") == "hierarchical_fusion":
        return HierarchicalFusionModel(config)
    else:
        return SimpleFusionModel(config)
