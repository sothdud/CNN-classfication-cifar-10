# src/utils.py
import matplotlib.pyplot as plt
import os

def plot_and_save(history, output_dir):
    # Loss
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    loss_plot_path = os.path.join(output_dir, 'loss_plot100.png')
    plt.savefig(loss_plot_path)
    plt.close()

    # Accuracy
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    acc_plot_path = os.path.join(output_dir, 'accuracy_plot100.png')
    plt.savefig(acc_plot_path)
    plt.close()
