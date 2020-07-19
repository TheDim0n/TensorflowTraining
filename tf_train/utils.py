from tensorflow import argmax


__all__ = ['predict_classes']

def predict_classes(model, features, training=False):
    """
    Predict classes for feature list.

    Args:
        model: Keras model that predicts.

        features: Feature list.

        training: If True, model trainable variables will train. Default False.
    """
    return argmax(model(features, training=training), axis=-1)


if __name__ == "__main__":
    pass
