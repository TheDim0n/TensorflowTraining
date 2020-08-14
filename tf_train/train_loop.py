from tensorflow import GradientTape
from tensorflow.keras import metrics, optimizers
import time

from . import api
from .utils import predict_classes


__all__ = ['train', 'evaluate']

def evaluate(dataset, model, loss, metrics=[metrics.Recall(), metrics.Precision()]):
    _data = api.evaluate(dataset=dataset, model=model, loss=loss, metrics=metrics)
    for name, val in _data.items():
        print('{}: {:.6f}'.format(name, val), end='')
    print()


def train(train_dataset, model, epochs, loss, optimizer=optimizers.Adam(), metrics=[metrics.Recall(), metrics.Precision()], frequency=10, validation_dataset=None):
    """
    Train keras model.

    Args:
    
        train_dataset: Tensorflow Dataset object for train the model.

        model: Keras trainable model.

        epochs (int): Num of epochs of training.

        loss: Loss function.

        optimizer: Tensorflow optimizer. Default Adam with learning_rate=0.001.

        metrics: List of tensorflow mertics. Default contains Recall and Precision.
    """

    model.compile(optimizer=optimizer, loss=loss)

    for metric in metrics:
        metric.reset_states()

    __start_train_time = time.time()
    __start_epoch_time = __start_train_time

    for epoch in range(epochs + 1):

        for x, y in train_dataset:
            grad = api.__grad(model, x, y, loss=loss)
            loss_value, grads = grad['loss_value'], grad['grad_tape']
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            #update metrics values
            for metric in metrics:
                y_pred = predict_classes(model, x, training=True)
                metric.update_state(y_true=y, y_pred=y_pred)

        if epoch % frequency == 0:
            print("Epoch {} -- Loss({}): {:.6f}".format(epoch, api.__get_name(loss)['name'], loss_value), end='')
            for metric in metrics:
                print(', {}: {:.6f}'.format(api.__get_name(metric)['name'], metric.result()), end='')
            print(', time - {:.3f}s'.format(time.time() - __start_epoch_time))

            if validation_dataset:
                a = '_' * (len(str(epoch)) + 9)
                print(a, end='')
                evaluate(validation_dataset, model, loss, metrics=metric)

            __start_epoch_time = time.time()

    print('Model training takes {:.3f} seconds.'.format(time.time() - __start_train_time))

            
if __name__ == "__main__":
    pass
