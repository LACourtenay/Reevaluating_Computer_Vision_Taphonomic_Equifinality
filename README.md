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
    * Code used to conduct quality assessment of each image in each of the datasets, additionally producing gradient maps and supplementary images to help assess image quality. The output of much of this code is available at: https://doi.org/10.6084/m9.figshare.24877743.v3 . Numeric results are included in the Results folder of this same repository.
  * <b>Preliminary Experiment.ipynb</b>
    * Jupyter notebook demonstrating some of the concepts of how to evaluate image quality using the computer vision codes. This includes an example of an archaeological cut mark photographed under different conditions and then studied with each of the techniques, plus experiments gradually blurring the image and observing how this effects the numeric results produced throughout analyses
  * <b>Results Folder</b>
    * This contains the numeric results and output produced by each of the codes provided within this folder. All visual maps, representations and figures were too large to include in GitHub and are therefore stored and available from: https://doi.org/10.6084/m9.figshare.24877743.v3 
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
* <b> Greyscale_Examples </b>
  * <b>Greyscale_Comparison_Code.ipynb </b>
    * Supplementary code used to assess the differences between RGB images, and greyscale images that either have 1 or 3 channels, as well as how these conversions can be performed. This file uses the three example images provided in the same folder, descriptions of which are provided in the jupyter notebook.
* <b> DS1_File_Index.txt </b>, <b> DS2_File_Index.txt </b> and <b> DS3_File_Index.txt </b> file
  * Files detailing the contents of each of the folders, also indicating which images belong to which classes.
  
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

<b>NOTE: </b> All code was run using the CPU of a basic laptop. No GPU was used. This can result in relatively long training times. If the user wishes to use GPU, please adapt the code accordingly. Likewise, all code was run on a local machine, no cloud computation was employed, if the user wishes to run code in the cloud, or using an external server or service, such as google colab, they may need to adapt their code to allow and facilitate this.

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

For all applications, data must be sorted into a folder called "Train" within either "DS1", "DS2", or "DS3". Inside the train folder, the user needs to sort the images also into their corresponding class folders, e.g. in DS1 the Train folder must contain a folder containing the crocodile tooth marks, and one containing the cut marks. The code will then sort the data from the train folder into either the test or validation folder, according to the train:test:validation split.

The general structure of the directory should therefore be as follows, before executing any of the code:

```
./
  ├── CodeToRun.py
  ├── DS1/
  │   ├── Train/
  │   │   ├── CutMark/
  │   │   ├── Crocodile/
  ├── DS2/
  │   ├── Train/
  │   │   ├── CutMark/
  │   │   ├── Score/
  │   │   ├── Trampling/
  ├── DS3/
  │   ├── Train/
  │   │   ├── CutMark/
  │   │   ├── Score/
  │   │   ├── Trampling/
```

The original datasets, in most cases, are not clean at all (DS3 being the main exception, DS2 presents a brief description in the abstract of the dataset), and it may not always be clear what images pertain to which class. Files with names containing the characters 'SF' are cut marks, the term SF refering to 'Simple Flake'. The characters 'LS', or direct names of animals such as 'wolves', 'bears', 'lions' or 'hyenas', evidently refer to tooth scores. Trampling marks are files containing either the word 'trampling', or the characters 'tmp'. Finally crocodile tooth scores contain 'CS' (Crocodile Score) in the file name. For DS3, the user should only download the experimental data, not the archaeological data, so one folder should be excluded, while the dataset should also be fused with DS2 (as explained in our manuscript, and the original publication of DS3). The three .txt files containing the term File_Index in the name provide a detailed index of which folders the images should be placed in.

--------------------------------------------------------

### <b> First round of reviews and changes to code  [*] </b>

Thanks to the suggestions of 3 very constructive and helpful reviewers, changes were made to parts of the analysis performed which has an impact on the jupyter notebook files contained within the folder <b> Deep Learning Code </b>. These changes involve the removal of part of the analysis, as suggested by the reviewers. The original manuscript contained comments regarding the performance of algorithms when attacked by adversarial noise. As correctly pointed out by reviewers, however, adversarial noise is carefully calculated and not a true reflection of what would happen if algorithms are exposed to natural random noise. For this reason, the final section regarding adversarial noise was removed from the revised version of the manuscript. HOWEVER, because the objective of uploading jupyter notebooks presenting the trained CNNs was to ensure transparency, and show the reader the precise output of each file without doctoring or modifying the file afterwards, this presents some issues that would either require completely re-running all of the codes, which would then require updating the entire results section of the manuscript, or deleting cells, which would then become apparent or look like we are trying to hide things. For this reason, our goal is to ensure that the jupyter notebooks are non-adultered, and have not been touched since being run, and instead we will provide the following statements identifying which cells of code for each file are to be discarded as no longer forming part of the study.

* <b>DS1 Final Training Jason2.ipynb</b>
  * Cell 11 - contains code that extracts the real labels of each image from the training generator as well as calculating the labels and probabilities predicted by the algorithm without the introduction of adversarial noise, called original_probs, original_labels and real_labels. The definition and contents of these three objects are fine, all other objects and calculations must be discarded.
  * Cell 13 and 14 - contains the comparison of loss and confidence probabilities when making predictions for both adversarial examples, and the algorithms before adversarial noise are introduced. All adversarial examples must be discarded.
  * Cell 15 - contains loss rates of the algorithms before (original) and after (adversarial) adversarial noise is introduced into the equation, all adversarial examples must be discarded.
  * Cell 16 - must be discarded completely
  * Cell 18 and 19 - please ignore the green curves in both figures.
* <b>DS2 Final Training DenseNet201.ipynb</b>
  * Cell 10 - contains code that extracts the real labels of each image from the training generator as well as calculating the labels and probabilities predicted by the algorithm without the introduction of adversarial noise, called original_probs, original_labels and real_labels. The definition and contents of these three objects are fine, all other objects and calculations must be discarded.
  * Cell 13 and 14 - contains the comparison of loss and confidence probabilities when making predictions for both adversarial examples, and the algorithms before adversarial noise are introduced. All adversarial examples must be discarded.
  * Cell 15 - contains loss rates of the algorithms before (original) and after (adversarial) adversarial noise is introduced into the equation, all adversarial examples must be discarded.
  * Cell 16 - must be discarded completely
* <b>DS3 Final Training DenseNet201.ipynb</b>
  * Cell 11 - contains code that extracts the real labels of each image from the training generator as well as calculating the labels and probabilities predicted by the algorithm without the introduction of adversarial noise, called original_probs, original_labels and real_labels. The definition and contents of these three objects are fine, all other objects and calculations must be discarded.
  * Cell 14 and 15 - contains the comparison of loss and confidence probabilities when making predictions for both adversarial examples, and the algorithms before adversarial noise are introduced. All adversarial examples must be discarded.
  * Cell 16 - contains loss rates of the algorithms before (original) and after (adversarial) adversarial noise is introduced into the equation, all adversarial examples must be discarded.
  * Cell 17 and 19 - must be discarded completely

It is also important, therefore, to point out that the original version of the figshare repository (https://doi.org/10.6084/m9.figshare.24877743.v1) contains code and figures regarding these adversarial attack analyses, the revised version which is now published as version 3 at https://doi.org/10.6084/m9.figshare.24877743.v3, does not contain these files. In this repository, however, you will find the Grad-CAM results for the entire study in a zip file that contains the original image, the Grad-CAM of this original image, and finally the Grad-CAM of the image after introducing adversarial noise, as stated before, this third image should therefore be disregarded since the first review of the present study.

--------------------------------------------------------

## <b> Repository Citation </b>

Please cite this repository as:

 <b> Courtenay, L.A. (2024) Code for the reevaluation of deep learning based computer vision algorithms for the classificaiton of taphonomic Bone Surface Modifications https://github.com/LACourtenay/Reevaluating_Computer_Vision_Taphonomic_Equifinality </b>

--------------------------------------------------------

Comments, questions, doubts, suggestions and corrections can all be directed to L. A. Courtenay at the email provided above.
