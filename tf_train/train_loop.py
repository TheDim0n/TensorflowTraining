from tensorflow import GradientTape
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
import time

from .utils import predict_classes


__all__ = ['train', 'evaluate']

def __get_name(obj):
    return str(obj).split()[0].split('.')[4]

def __calculate_loss(model, x, y, loss, training=True):
    y_pred = model(x, training=training)
    return loss(y_true=y, y_pred=y_pred)

def __grad(model, x, y, loss):
    with GradientTape() as tape:
        loss_value = __calculate_loss(model, x, y, loss)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def evaluate(dataset, model, loss, metrics=[metrics.Recall(), metrics.Precision()]):
    for x, y in dataset:
        loss_value = __calculate_loss(model, x, y, loss=loss, training=False)
        for metric in metrics:
            y_pred = predict_classes(model, x, training=False)
            metric.update_state(y_true=y, y_pred=y_pred)

    print("Loss({}): {:.6f}".format(__get_name(loss), loss_value), end='')
    for metric in metrics:
        print(', {}: {:.6f}'.format(__get_name(metric), metric.result()), end='')
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
            loss_value, grads = __grad(model, x, y, loss=loss)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            #update metrics values
            for metric in metrics:
                y_pred = predict_classes(model, x, training=True)
                metric.update_state(y_true=y, y_pred=y_pred)

        if epoch % frequency == 0:
            print("Epoch {} -- Loss({}): {:.6f}".format(epoch, __get_name(loss), loss_value), end='')
            for metric in metrics:
                print(', {}: {:.6f}'.format(__get_name(metric), metric.result()), end='')
            print(', time - {:.3f}s'.format(time.time() - __start_epoch_time))

            if validation_dataset:
                print('*', ' ' * (len(str(epoch)) + 8), end='')
                evaluate(validation_dataset, model, loss, metrics)

            __start_epoch_time = time.time()

    print('Model training takes {:.3f} seconds.'.format(time.time() - __start_train_time))

            
if __name__ == "__main__":
    pass
