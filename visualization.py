import matplotlib.pyplot as plt


def plot_training_history(history):
    plt.figure()
    training_loss = history["training_loss"]
    validation_loss = history["validation_loss"]
    validation_loss_best = history["validation_loss_best"]

    plt.plot(training_loss, label="Training Loss", linestyle="-", marker="o")
    plt.plot(validation_loss, label="Validation Loss", linestyle="--", marker="s")
    plt.plot(validation_loss_best, label="Best Validation Loss", linestyle=":", marker="^")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)
    plt.show()