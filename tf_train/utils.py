from tensorflow import argmax


__all__ = ['predict_classes']

def predict_classes(model, features, training=False):
    return argmax(model(features, training=training), axis=-1)


if __name__ == "__main__":
    pass
