# Reevaluating_Computer_Vision_Taphonomic_Equifinality
Code used for the evaluation of how reliable deep learning based computer vision algorithms are for the study of taphonomic equifinality. This code was used to carry out the study titled <b> Deep Learning-based Computer Vision is not yet the answer to taphonomic equifinality in Bone Surface Modifications </b>, by Lloyd A. Courtenay, Nicolas Vanderesse, Luc Doyon and Antoine Souron, currently under review at the <i> Journal of Computational Applications in Archaeology </i>.

-----------------------------------------------------------------------------------------------------------------

## <b> Author Details </b>

<b> Author </b>: Lloyd A. Courtenay

<b> Email </b>: ladc1995@gmail.com

<b> ORCID </b>: https://orcid.org/0000-0002-4810-2001

<b> Current Afiliation </b>: University of Bordeaux [CNRS, PACEA UMR5199]

---------------------------------------------------------------------------------------------------

This code has been designed for the open-source free Python programming languages.

---------------------------------------------------------------------------------------------------

## <b> Repository Details </b>

The present repository contains:

* <b> Computer Vision Analysis Code </b>
  * <b>Analysis of Images According to Sample.ipynb</b>
    * Jupyter notebook containing the code used to assess the quality of images in each of the datasets, separating between what the images are of (the sample)
  * <b>Computer Vision Analysis.py</b>
    * Code used to conduct quality assessment of each image in each of the datasets, additionally producing gradient maps and supplementary images to help assess image quality. The output of much of this code is available at: https://doi.org/10.6084/m9.figshare.24877743.v1 . Numeric results are included in the Results folder of this same repository.
  * <b>Preliminary Experiment.ipynb</b>
    * Jupyter notebook demonstrating some of the concepts of how to evaluate image quality using the computer vision codes. This includes an example of an archaeological cut mark photographed under different conditions and then studied with each of the techniques, plus experiments gradually blurring the image and observing how this effects the numeric results produced throughout analyses
  * <b>Results Folder</b>
    * This contains the numeric results and output produced by each of the codes provided within this folder. All visual maps, representations and figures were too large to include in GitHub and are therefore stored and available from: https://doi.org/10.6084/m9.figshare.24877743.v1 
* <b> Deep Learning Code </b>
  * <b>DS1 Final Training Jason2.ipynb</b>
    * Jupyter notebook with python code demonstrating the training of Jason2 for classification tasks on DS1 (see below for information about the datasets) [*]
  * <b>DS2 Final Training VGG16.ipynb</b>
    * Jupyter notebook with python code demonstrating the training of VGG16 for classification tasks on DS2 (see below for information about the datasets) [*]
  * <b>DS3 Final Training DenseNet201.ipynb</b>
    * Jupyter notebook with python code demonstrating the training of DenseNet201 for classification tasks on DS3 (see below for information about the datasets) [*]
  * <b>Example of Hypothetical Learning Curves.py</b>
    * Python code used to generate synthetic data and train a simple neural network under three forced conditions; overfitting, underfitting, and perfect fitting to the data.
  * <b>Grad_CAM_Code.py</b>
    * Python code used to perform Grad-CAM analyses on each of the datasets. This code is run using a agrument parser in command prompt. Simply run using the following prompt in an activated and appropriate conda environment;
      * <b> python Grad_CAM_Code.py --dataset 1 </b> - <i> for dataset 1 </i>
      * <b> python Grad_CAM_Code.py --dataset 2 </b> - <i> for dataset 2 </i>
      * <b> python Grad_CAM_Code.py --dataset 3 </b> - <i> for dataset 3 </i>
  
[*] See additional comments bellow on the removal of portions of the code due to the reviewing process.

--------------------------------------------------------

## <b> System Requirements for Deep Learning </b>

* Python
    * Version 3.0 or higher
* Tensorflow
    * Version 2.0 or higher
* Numpy

--------------------------------------------------------

## <b> Citation </b>

Please cite this repository as:

 <b> Courtenay, L.A. (2024) Code and Data for the Modelling of Pachycrocuta tooth pits using Variational Autoencoders and Markov Chain Monte Carlo algorithms. https://github.com/LACourtenay/VAE_MCMC_Pachycrocuta_Simulations </b>

--------------------------------------------------------

Comments, questions, doubts, suggestions and corrections can all be directed to L. A. Courtenay at the email provided above.

### How to structure data:

For computer vision analysis data must simply be in a folder called DS1, DS2 or DS3. For deep learning applications, data must be in a folder called "Train" within either "DS1", "DS2", or "DS3". The code will then sort the data from the train folder into either the test or validation folder.