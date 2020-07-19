from tensorflow import GradientTape
from tensorflow.keras import optimizers
from tensorflow.keras import metrics

from .utils import predict_classes


__all__ = ['train']


def train(train_dataset, model, epochs, loss, optimizer=optimizers.Adam(), metrics=[metrics.Recall(), metrics.Precision()]):
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

    def __get_name(obj):
        return str(obj).split()[0].split('.')[4]

    def __calculate_loss(model, x, y, training=True):
        y_pred = model(x, training=training)
        return loss(y_true=y, y_pred=y_pred)

    def __grad(model, x, y):
        with GradientTape() as tape:
            loss_value = __calculate_loss(model, x, y)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)


    for epoch in range(epochs + 1):

        for x, y in train_dataset:
            loss_value, grads = __grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            #update metrics values
            for metric in metrics:
                y_pred = predict_classes(model, x, training=True)
                metric.update_state(y_true=y, y_pred=y_pred)

        if epoch % 10 == 0:
            print("Epoch {:03d} -- Loss({}): {:.3f}".format(epoch, __get_name(loss), loss_value), end='')
            for metric in metrics:
                print(', {}: {}'.format(__get_name(metric), metric.result()), end='')
            print()

            
if __name__ == "__main__":
    pass
