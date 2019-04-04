# Team-Challenge
Team challange group 1
Writen by: Colin Nieuwlaat, Jelle van Kerkvoorde, Mandy de Graaf, Megan Schuurmans & Inge van der Schagt

# General description:
    This program performs segmentation of the left vertricle, myocardium, right ventricle
    and backgound of Cardiovascular Magnetic Resonance Images, with use of a convolutional Unet based
    neural network. From each patient,  both a 3D end systolic image and a 3D end diastolic image is 
    with its ground truth is available. The data set is devided into a training set, validation set and a test set. 
    
    First, The images are obtained from the stored location. Subsequently, these images are 
    pre-processed which includes normalisation, removal of outliers and cropping. The training
    subset is used to train the Network, afterwards, the Network is validated and further trained
    by using the validation subset.
    
    After the network is trained, the network is evaluated using the subset regarding testing. 
    The test images are segmented, using the trained network and these segmentations are 
    evaluated by compairing them to the ground truth. The Dice Coefficient is calculated
    to evaluate the overlay of the segmentation and the ground truth. Furthermore, the 
    Hausdorff Distance is computed. 
    
    From the segmentations of the left ventricular cavity during the end systole and
    end diastole, the ejection fraction is calculated. This value is compared to
    the computed ejection fraction calculated from the ground truth.

# Contents program:
    - TC_main.py:  Current python file, run this file to run the program.
    - TC_model.py: Contains functions that initializes the network.
    - TC_data.py:  Contains functions that initializes the data, preprocessing and 
                   metrics used in training, evaluation & testing.
    - TC_test.py:  Contains functions that show results of testing. 
    - TC_visualization.py: visualises the intermediated and final results.
    - TC_helper_functions.py: contains functions to make the main more clean
    - Data: a map with all the patient data.
    
    
# Variables:
    
    trainnetwork:       Can be set to True or False. When set to True, the network
                        is trained. When set to False, a Network is loaded from the
                        networkpath.
    evaluatenetwork:    Can be set to True or False. When set to True, the network is
                        evaluated. If set to False, no evaluation is performed
    networkpath:        Path to the stored Network 
    trainingsetsize:    Number between 0 and 1 which defines the fraction of the data
                        that is used for training. 
    validationsetsize:  Number between 0 and 1 which defines the fraction of the 
                        training set that will be used for validation.
                        
    num_epochs:         Integer that defines the number of itarations. Should be increased
                        when the network should train more and should be decreased when
                        the network does not learn any more.
       

    dropout:            Can be set to True or False in order to involve Drop-out
                        in the Network or not.
    dropoutpct:         Float between 0 and 1 which defines the amount of Drop-out
                        you want to use. The higher the value, the more feature maps
                        are removed         

    lr:                 Float which defines the initial learning rate. Should be increased 
                        when decreases very slowly.
    momentum:           COLIN KUN JIJ DIT UITLEGGEN? :)
    nesterov:           Can be set to True or False.
    
    
    
# Python external modules installed (at least version):
    - glob2 0.6
    - numpy 1.15.4
    - matplotlib 3.0.1
    - keras 2.2.4
    - SimpleITK 1.2.0
    - scipy 1.1.0

# How to run:
    Places all the files of zip file in the same map. 
    Make sure all modules from above in you python interpreter.
    Run TC_main in a python compatible IDE.
    If you want to train your network, set  trainnetwork to True in main()
    If you want to evaluate your network, set evaluationnetwork to True in main()
    (you can find these at global settings).

