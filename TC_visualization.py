import matplotlib.pyplot as plt
import numpy as np

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

def visualize3Dimage(seg, im):
    # seg (10, 144, 144, 4)
    # im (10, 144, 144)
    plt.figure()

    # only one label
    seg0 = seg[:,:,:,0]
    seg1 = seg[:,:,:,1]
    seg2 = seg[:,:,:,2]
    seg3 = seg[:,:,:,3]

    n_rows = np.ceil(np.sqrt(im.shape[0]))
    n_cols = np.ceil(np.sqrt(im.shape[0]))

    for z in range(im.shape[0]):
      plt.subplot(n_rows, n_cols, 1 + z)
      plt.imshow(im[z, :, :], clim=(np.min(im), np.max(im)), cmap='gray')
      plt.imshow(np.ma.masked_where(seg0[z,:,:]!=0, seg0[z,:,:]==0), alpha=0.6, cmap='Oranges', clim=(0, 1))
      plt.imshow(np.ma.masked_where(seg1[z,:,:]!=1, seg1[z,:,:]==1), alpha=0.6, cmap='Reds', clim=(0, 1))
      plt.imshow(np.ma.masked_where(seg2[z,:,:]!=1, seg2[z,:,:]==1), alpha=0.6, cmap='Greens', clim=(0, 1))
      plt.imshow(np.ma.masked_where(seg3[z,:,:]!=1, seg3[z,:,:]==1), alpha=0.6, cmap='Blues', clim=(0, 1))

      plt.title('Slice {}'.format(z + 1))
