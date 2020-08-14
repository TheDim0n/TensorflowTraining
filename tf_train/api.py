from tensorflow import GradientTape
from tensorflow.keras import metrics, optimizers
import time

from .utils import predict_classes

__all__ = ['evaluate']


def __get_name(obj):
    return {
        'name': str(obj).split()[0].split('.')[4],
    }

def __calculate_loss(model, x, y, loss, training=True):
    y_pred = model(x, training=training)
    return {
        'loss': loss(y_true=y, y_pred=y_pred),
    }

def __grad(model, x, y, loss):
    with GradientTape() as tape:
        loss_value = __calculate_loss(model, x, y, loss)['loss']
        return {
            'loss_value': loss_value, 
            'grad_tape': tape.gradient(loss_value, model.trainable_variables),
        }

def evaluate(dataset, model, loss, metrics=[metrics.Recall(), metrics.Precision()]):
    for x, y in dataset:
        loss_value = __calculate_loss(model, x, y, loss=loss, training=False)['loss_value']
        for metric in metrics:
            y_pred = predict_classes(model, x, training=False)
            metric.update_state(y_true=y, y_pred=y_pred)
    result = {
        'loss_value': loss_value,
    }
    for metric in metrics:
        result[__get_name(metric)['name']] = metric.result()
    return result


if __name__ == "__main__":
    pass
