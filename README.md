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
      * <b> python Grad_CAM_Code.py --dataset 1 </b> <i> for dataset 1 </i>
      * <b> python Grad_CAM_Code.py --dataset 2 </b> <i> for dataset 2 </i>
      * <b> python Grad_CAM_Code.py --dataset 3 </b> <i> for dataset 3 </i>
  * <b>Failed DS1, DS2 and DS3 Folders</b>
    * All of the jupyter notebooks for the remaining algorithms trained on each of the datasets, including in many cases annotations as well. For DS1 this includes DenseNet201, EficientNetB7, ResNet50 and VGG16. For DS2 this includes Alexnet, DenseNet201, InceptionV3, Jason, Jason2, ResNet50. For DS3 this includes ResNet50 and VGG19
* <b> Ensemble Learning Preparation </b>
  * <b>Train Base Models v1.py</b>
    * Code used to train the original first-level learner models that were then implemented in an Ensemble Learning training strategy in Supplementary File 2 of the main article.
  * <b>Train Base Models v2.py</b>
    * Code used to train the original first-level learner models that were then implemented in an Ensemble Learning training strategy in Supplementary File 2 of the main article. In comparison to v1 of this same file, the difference is that this file performs a slightly different train, test, metatrain and validation split. THis is documented in detail in Supplementary File 2.
  
[*] See additional comments bellow on the removal of portions of the code due to the reviewing process.

--------------------------------------------------------

## <b> System Requirements for Deep Learning </b>

<i> Note that here we specify the versions of the libraries that we used for the present study. We are unaware if earlier or later versions of the same libraries will work or present problems, because we have not tested this, the objective here is simply to state how we ran each of the codes presented </i>

* Anaconda - <i> v.2.4.0 </i>
* Python - <i> v.3.10.9 </i>
* The following Python libraries
  * Tensorflow - <i>v.2.12.0</i>
  * Scikit Learn - <i>v.1.3.1</i>
  * Numpy - <i>v.1.22.0</i>
  * Matplotlib - <i>v.3.7.0</i>
  * Scipy - <i>v.1.7.3</i>
  * SciKit Image - <i>v.0.19.3</i>
  * OpenCV - <i>v.4.7.0</i>

All code was run using the CPU of a basic laptop

--------------------------------------------------------

### <b> Datasets and Directory Structure </b>

Due to issues related with intelectual property, the current authors do not have permission to share the datasets used in this study, as they are evidently not property of the current authors. The original datasets can be obtained from the following links:

* <b> DS1 </b>
  * Original paper - https://doi.org/10.1016/j.geobios.2022.07.001
  * Original dataset - https://doi.org/10.7910/DVN/9NOD8W
* <b> DS2 </b>
  * Original paper - https://doi.org/10.1038/s41598-020-75994-7
  * Original dataset - https://doi.org/10.7910/DVN/62BRBP
* <b> DS3 </b>
  * Original paper - https://doi.org/10.1080/08912963.2023.2242370
  * Original dataset - https://doi.org/10.17632/3bm34bp6p4.1




For computer vision analysis data must simply be in a folder called DS1, DS2 or DS3. For deep learning applications, data must be in a folder called "Train" within either "DS1", "DS2", or "DS3". The code will then sort the data from the train folder into either the test or validation folder.

--------------------------------------------------------

## <b> Citation </b>

Please cite this repository as:

 <b> Courtenay, L.A. (2024) Code for the reevaluation of deep learning based computer vision algorithms for the classificaiton of taphonomic Bone Surface Modifications https://github.com/LACourtenay/Reevaluating_Computer_Vision_Taphonomic_Equifinality </b>

--------------------------------------------------------

Comments, questions, doubts, suggestions and corrections can all be directed to L. A. Courtenay at the email provided above.