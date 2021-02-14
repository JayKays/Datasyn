import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    batch_size = 32
    neurons_per_layer = [64] * 1 + [10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    #Set main tricks to be tested in both models
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    learning_rate = 0.02 if use_momentum else .1 #Adjusting learning rate for momentum

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)


    # Creating comparison model: change True -> False to turn off tricks from prev. model

    use_momentum = use_momentum and True
    use_improved_sigmoid = use_improved_sigmoid and True
    use_improved_weight_init = use_improved_weight_init and True

    learning_rate = 0.02 if use_momentum else .1 #Adjusting learning rate for momentum

    model_no_shuffle = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_no_shuffle, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_no_shuffle, val_history_no_shuffle = trainer_shuffle.train(
        num_epochs)

    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Improved", npoints_to_average=10)
    utils.plot_loss(
        train_history_no_shuffle["loss"], "Old", npoints_to_average=10)
    plt.ylim([0, .4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, 1])
    utils.plot_loss(val_history["accuracy"], "Improved")
    utils.plot_loss(
        val_history_no_shuffle["accuracy"], "Old")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()
