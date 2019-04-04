import matplotlib.pyplot as plt
import numpy as np

def visualizeTraining(hist, dimensions, num_epochs, batchsize, nr_channels_start, nr_channels_end):
    '''
    Plots the training & validation softdice values and 
    plot training & validation loss values
    
    Parameters:
        hist:A History object. Its History.history attribute is a record of 
             training loss values and metrics values at successive epochs,as well
             as validation loss values and validation metrics values.
        dimensions: an int, that describes the dimensions of the images used. 2 for slices; 3 for entire 3d images
        num_epochs: ant int, that describes the number of epoches used
        batchsize: an int, that describes the size of the batch
        nr_channels_start: an int, that describes the number of channels at the start of the network
        nr_channels_end: an int, that describes the number of channels at the end of the network
    
    Returns:
        None
        
    '''
    
    # Plot training & validation softdice values
    plt.figure()
    plt.plot(hist.history['softdice_coef_multilabel'])
    plt.plot(hist.history['val_softdice_coef_multilabel'])
    plt.title('Model multilabel softdice')
    plt.ylabel('Multilabel softdice')
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

def visualize3Dimage(seg, im, savepath):
    '''
    Plots every slice of im with an overlay of the labels from seg
    
    Parameters:
        seg: a ndarray, segmetation images with 4 labels; 0 background (orange), 
             1 right ventricle (red) , 2 myocardium (green) & 3 left ventricle(blue)  
        im: a ndarray, the actual images of the ED or ES
        savepath: a string, that describes the path where final figure is saved
    
    Returns:
        None
    '''
    fig = plt.figure()

    # only one label
    seg0 = seg[:,:,:,0]
    seg1 = seg[:,:,:,1]
    seg2 = seg[:,:,:,2]
    seg3 = seg[:,:,:,3]

    n_rows = np.ceil(np.sqrt(im.shape[0]))
    n_cols = np.ceil(np.sqrt(im.shape[0]))
    
    #plots for ever slices the overlay
    for z in range(im.shape[0]):
        plt.subplot(n_rows, n_cols, 1 + z)
        plt.imshow(im[z, :, :], clim=(np.min(im), np.max(im)), cmap='gray')
        plt.imshow(np.ma.masked_where(seg0[z,:,:]!=0, seg0[z,:,:]==0), alpha=0.6, cmap='Oranges', clim=(0, 1))
        plt.imshow(np.ma.masked_where(seg1[z,:,:]!=1, seg1[z,:,:]==1), alpha=0.6, cmap='Reds', clim=(0, 1))
        plt.imshow(np.ma.masked_where(seg2[z,:,:]!=1, seg2[z,:,:]==1), alpha=0.6, cmap='Greens', clim=(0, 1))
        plt.imshow(np.ma.masked_where(seg3[z,:,:]!=1, seg3[z,:,:]==1), alpha=0.6, cmap='Blues', clim=(0, 1))
        plt.title('Slice {}'.format(z + 1))

    fig.savefig(savepath)
    plt.close(fig)
