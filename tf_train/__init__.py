from tensorflow.keras import models
from tensorflow import GradientTape
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics


def train(train_dataset, model, epochs, loss: losses.Loss, optimizer=optimizers.Adam, metrics=[metrics.Recall, metrics.Precision]):
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
    
    def __calculate_loss(x, y, training=True):
        y_pred = model(x, training=training)
        return loss(y_true=y, y_pred=y_pred)

    def __grad(x, y):
        with GradientTape as tape:
            loss_value = __calculate_loss(x, y)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)


    for epoch in range(epochs + 1):

        for x, y in train_dataset:
            loss_value, grads = __grad(x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            #update metrics values
            for metric in metrics:
                metric.update_state(y, model(x, training=True))

        if epoch % 100 == 0:
            print("Epoch {:03d} -- Loss({}): {:.3f}".format(epoch, loss.__name__, loss_value), end='')
            for metric in metrics:
                print(', {}: {}'.format(metric.__name__, metric.result()), end='')
            print()

            
if __name__ == "__main__":
    pass