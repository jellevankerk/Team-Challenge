import matplotlib.pyplot as plt

def visualizeTraining(hist, dimensions, num_epochs, batchsize, nr_channels_start, nr_channels_end):
    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(hist.history['softdice_coef_multilabel'])
    plt.plot(hist.history['val_softdice_coef_multilabel'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(r'Plots/accplot_{}D_epochs={}_bs={}_channels={}-{}.png'.format(dimensions, num_epochs, batchsize, nr_channels_start, nr_channels_end))

    # Plot training & validation loss values
    plt.figure()
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(r'Plots/lossplot_{}D_epochs={}_bs={}_channels={}-{}.png'.format(dimensions, num_epochs, batchsize, nr_channels_start, nr_channels_end))
