from src.models.swin_transformer import SwinTransformer

class ModelFactory:
    """
    Returns the classes for models.
    """
    MODEL_CLASSES = {
        "swin_transformer": SwinTransformer
    }
