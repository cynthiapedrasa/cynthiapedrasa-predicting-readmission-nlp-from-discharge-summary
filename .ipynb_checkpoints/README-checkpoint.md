# Data Science Capstone Project
Student name: Cynthia Pedrasa

##  **Predicting 30-Day All-Cause Readmission from Hospital Discharge Summary**



<center><img src="images/HospReadmissions.png" alt="drawing" width="700"/></center>  


## Hospital Readmissions are both a Clinical and Financial Problem!

* [CMS began penalizing hospitals for 30-day readmissions Oct. 1, 2012 at 1 percent, upping the penalty rate to 2 percent for fiscal year 2014](https://www.beckershospitalreview.com/quality/6-stats-on-the-cost-of-readmission-for-cms-tracked-conditions.html) 
* [CMS will cut payments to the penalized hospitals by as much as 3 percent for each Medicare case during fiscal 2020, which runs Oct. 1 through September 2020](https://www.beckershospitalreview.com/finance/cms-penalizes-2-583-hospitals-for-high-readmissions-5-things-to-know.html)
* [All-cause readmissions - The average cost of a readmission for any given cause is **$11,200**, with a 21.2 percent readmission rate](https://www.beckershospitalreview.com/quality/6-stats-on-the-cost-of-readmission-for-cms-tracked-conditions.html)  


## Business Drivers

<center><img src="images/BusinessDrivers.png" alt="drawing" width="500"/></center>  



Currently, clinical data use is limited to the structured information.  Dashboards are limited to reporting discrete data elements and coded information.  However, it was reported that [more than 80 percent of a healthcare organization's data is unstructured](https://www-03.ibm.com/press/us/en/pressrelease/42179.wss), including physician notes, clinical assessments, registration forms, discharge summaries and other nonstandardized electronic forms, which makes data collection and analysis difficult using standard methods.  Insights we could get from the doctors notes that are free text in nature and if we are able to identify risk factors from the sea of data, we might be able to supplement prediction of readmission risk and improve outcomes for the patients.


<center><img src="images/TypesofData.png" alt="drawing" width="800"/></center>
 
                                    Source: Weber GM, et al., Finding the missing link for big biomedical data. JAMA 2014; 311(24):

|Barriers to healthcare data:|
|:---|
|* patient protection|
|* data quality|
|* cost(monetary, time,resources)|
|* transparency|
|* disparate rules across stakeholders|

|Structured Data Characteristics|Unstructured Data Characteristics|
|:---|:---|
|Pre-defined ontology | Not pre-defined – may be text, image, sound, video |
|Easy to search| Difficult to search|
|Examples:| Examples: |
|ICD-10-CM| Discharge Summary |
|CPT | Clinical notes|
|LOINC| Radiographs|
|SNOMED | Mobile health data| 


We will use natural language processing to turn the unstructured discharge summary data into information that will help identify at-risk patients and allow the clinicians to intervene.  Hospital discharge summaries serve as the primary documents communicating a patient’s care plan to the post-hospital care team. The discharge summary is the form of communication that accompanies the patient to the next setting of care. High-quality discharge summaries are generally thought to be essential for promoting patient safety during transitions between care settings, particularly during the initial post-hospital period. It plays an important role in preventing avoidable hospital readmissions.

|The Joint Commission mandates that six components be present in all U.S. hospital discharge summaries: |
|:---|
|1. Reason for hospitalization|
|2. Significant findings| 
|3. Procedures and treatment provided| 
|4. Patient’s discharge condition|  
|5. Patient and family instructions (as appropriate)|  
|6. Attending physician’s signature| 


 <center><img src="images/DischargeSum.png" alt="drawing" width="600"/></center>

         

  
## Build the MIMIC Database

We will utilize the MIMIC-III (Medical Information Mart for Intensive Care III), a free hospital database. Mimic III is a relational database that contains de-identified data from over 40,000 patients who were admitted to Beth Israel Deaconess Medical Center in Boston, Massachusetts from 2001 to 2012. MIMIC-III contains detailed information regarding the care of real patients, and as such requires credentialing before access.

In order to get access to the data for this project, you will need to request access at this link (https://mimic.physionet.org/gettingstarted/access/) and complete the required training course at CITI “Data or Specimens Only Research" 


* [Register for the required training course](https://www.citiprogram.org/index.cfm?pageID=154&icat=0&ac=0) as “Massachusetts Institute of Technology" affiliate
* [Request access to the Mimic Database:](https://mimic.physionet.org/gettingstarted/overview/)
*  Download the full MIMIC-III dataset from: https://doi.org/10.13026/C2XW26
*  Access import.py at https://github.com/MIT-LCP/mimic-code/tree/master/buildmimic/sqlite to generate a SQLite database from the MIMIC-III csv files. 

from IPython.display import IFrame
IFrame('https://mit-lcp.github.io/mimic-schema-spy/relationships, width=700, height=450)

## Data Description

A SQLite database was generated using the MIMIC III CSV files. MIMIC-III is a relational database consisting of 26 tables. For a detailed description of the database structure, see the [MIMIC-III Clinical Database:](https://mimic.physionet.org/mimictables/) and [Database Schema:](https://mit-lcp.github.io/mimic-schema-spy/)


 <center><img src="images/MimicTables.png" alt="drawing" width="700"/></center>
 
The data files are distributed in comma separated value (CSV) format following the RFC 4180 standard. Notably, string fields which contain commas, newlines, and/or double quotes are encapsulated by double quotes ("). Actual double quotes in the data are escaped using an additional double quote. For example, the string `she said "the patient was notified at 6pm"` would be stored in the CSV as `"she said ""the patient was notified at 6pm"""`. More detail is provided on the RFC 4180 description page: https://tools.ietf.org/html/rfc4180

## Visualization
Many unstructured notes e.g. assessment, medical history, progress notes, discharge notes, etc. are generated daily by the multi-disciplinary healthcare workers. 
<center><img src="images/NotesCat.png" alt="drawing" width="500"/></center>

For Hospital Readmissions, we are only concerned about the adult and non-elective encounters.
<center><img src="images/AdmissionTypes.png" alt="drawing" width="500"/></center>

We are  predicting the presence of a readmission risk, for example, "yes" (1) would mean they have readmissions, and "no" (0) would mean they don't have the hospital readmissions.  <center><img src="images/ClassLabel.png" alt="drawing" width="400"/></center>               
30-day hospital readmission is a problem!                     <center><img src="images/30dayReadmit.png" alt="drawing" width="400"/></center> 

                           
What were the sources of admission for our patients?  <center><img src="images/AdmissionType.png" alt="drawing" width="800"/></center>


## Data Preprocessing

<center><img src="images/DataPrep.png" alt="drawing" width="700"/></center>

**TEXT PROCESSING**  
Note:  We skipped the use of Stemming (process of reducing each word to its root or base) and Lemmatization (a more calculated process of returning the base or dictionary form of a word) in this project.  
* Text cleaning is task specific – medical abbreviations.   
* Split discharge summary into words/tokens  
* Remove punctuation  
* Remove non-alphabetic characters  
* Remove new line/carriage return  
* Filter out tokens that are stop words i.e. the most common words that do not contribute to the deeper meaning of the phrase   


After cleaning the text and splitting them into tokens,  we converted the discharge summary text to numbers.  

* **Bag-of-Words Model (BoW)**  
We may want to perform classification of documents, so each discharge summary from the admission is an input and a class label (“Readmission”) is the output for our predictive algorithm. Algorithms take vectors of numbers as input, therefore we need to convert documents to fixed-length vectors of numbers. For this project we will be utilizing the Bag-of-Words Model, or BoW. This model doesn’t focus about the order of words but focuses on the occurrence of words in a document or the degree to which they are present in encoded.  

* **Count Vectorizer**  
Count Vectorizer will be utilized to tokenize a collection of text documents, build a vocabulary of known words and also to encode new documents using that vocabulary. 
An encoded vector is returned with a length of the entire vocabulary and an integer count for the number of times each word appeared in the document.
Here are the steps:

   1. Create an instance of the CountVectorizer class.  
   
   ```console
   vector = CountVectorizer(max_features = 3000, tokenizer = clean_tokenize, stop_words = stop_words) 
   ```
   
   2.  Call the fit() function in order to learn a vocabulary from one or more documents.  
   
    ```console
    vector.fit(df_train.TEXT.values)
     ```
     
   3.  Call the transform() function on one or more documents as needed to encode each as a vector.  
   
     ```console
     X_train_tf = vect.transform(df_train.TEXT.values) X_valid_tf = vect.transform(df_valid.TEXT.values)
      ```
      
* **TF-IDF - Term Frequency - Inverse Document Frequency** - word frequency scores that try to highlight words that are more frequent in a documents but not across documents.  
    +  Term Frequency:  summarizes how often a given word appears within a document
    +  Inverse Document Frequency:  downscales words that appear a lot across documents
    TF-IDF will tokenize documents, learn the vocabulary and inverse document frequency weightings, and also to encode new documents using that vocabulary

<center><img src="images/WordCloud.png" alt="drawing" width="800"/></center>


## Model Selection
Spot-check machine learning classification algorithms:

* Logistic Regression LR
* Decision Tree DT
* Random Forest RF
* Multinomial Naive Bayes MNB
* Adaboost AB
* XGBoost XGB  
* Support Vector Machine SVM  

<center><img src="images/CompareModels.png" alt="drawing" width="600"/></center>

XGBoost Classifier was selected as the model with the best AUC score to predict readmissions from discharge notes.

## Train Test Split
For cross-validation,the dataset was split into training, validation, and test sets.

Grid-search was run to determine the best parameters for the selected classifier. The hypertuned parameters were fit on the training data and assessed using the following metrics:

* Validation accuracy
* Confusion matrices
* ROC curves

<center><img src="images/Tuning.png" alt="drawing" width="1000"/></center>
### Cross-Validation: Validation Accuracy
A given classification model was fit on the training data. It then classified the validation data. To assess the accuracy of the model, those predictions were compared to the actual labels. 

## Confusion Matrices
A confusion matrix was generated for each classifier. The confusion matrix is used to show the number of:

* True positives (TP): These are cases in which we predicted yes - they have the readmit risk, and they were readmitted.
* True negatives (TN): We predicted not at risk, and they were not readmitted.
* False positives (FP): We predicted yes, but they don't actually have readmissions. (Also known as a "Type I error.")
* False negatives (FN): We predicted no, but they actually have readmission. (Also known as a "Type II error.")


## Performance Metrics 
<center><img src="images/ModelEval.png" alt="drawing" width="600"/></center>
* True Positive Rate (TPR) is the proportion of actual readmissions that the test correctly predicted would be readmitted (Sensitivity/Recall)  
* True Negative Rate (TNR) is the proportion of patients that are not readmitted that are correctly identified as such (Specificity)  
* False Positive Rate (FPR) is the proportion of patients whom the model predicted would be readmitted, but were not readmitted      
* ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. This curve plots the TPR vs. FPR at different classification thresholds. Lowering the classification threshold classifies more items as positive, thus increasing both False Positives and True Positives.  Classifiers that give curves closer to the top-left corner indicate a better performance. The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test.  
* AUC is a single number that can evaluate a model’s performance, regardless of the chosen decision boundary. AUC ranges in value from 0 to 1. 
A model whose predictions are 100% wrong has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0.
* Prevalence: How often does the yes condition actually occur in our sample?
* Precision = Positive Predictive Value - Tells how often is it correct when it predicts that the patient is at risk for readmisssion
* If we were to choose a boundary of .8, readmittance probability above .8 is a readmission, everyone below is not   

## Feature Importance

<center><img src="images/FeatureImportance.png" alt="drawing" width="800"/></center>

## Conclusions
+ BoW with XGBoost Classifier (unigram) was selected as the best model to predict readmissions from discharge notes.  
+ BoW with XGB model performed better than TF-IDF and the complex Neural Network model.
+ Increasing n-gram range did not improve scores for BoW method.
+ Train data overfitting - early stop and tuning the regularization parameter    
<center><img src="images/Classifier.png" alt="drawing" width="300"/></center>

## Future Work

Compared to random predictions, results from our predictive model (AUC=.71) is a good baseline for further improving our model.  
Feature engineering, ensemble of models and parameter tuning of the model will help the adoption of the model as a clinical decision system for evaluating readmission
Explore other unstructured notes and/or combine with structured clinical information to strengthen predictive scores.  

Predicting hospital readmissions based on unstructured data opens many opportunities in predictive analytics where a vast amount of untapped data could be utilized to reduce hospital readmissions, improve outcomes for the patients, lower healthcare cost while providing quality care.  

Reduce Readmission by Predicting Patients at risk for Readmission

<center><img src="images/ReduceReadmission.png" alt="drawing" width="800"/></center>



===========================================================================
### Prerequisites
===========================================================================  
You may need to install some software and packages.


1. Install Anaconda (https://docs.anaconda.com/anaconda/install/)

2. Install SQLite (https://sqlitebrowser.org/)

3. Install Scikit-learn (https://anaconda.org/anaconda/scikit-learn)
```console
conda install -c anaconda scikit-learn
```
4. Install NLTK (http://www.nltk.org/install.html)
```console
conda install -c anaconda nltk
```

5. Install the TensorFlow deep learning library 

6. Install Imbalanced-Learn Library (https://anaconda.org/conda-forge/imbalanced-learn)
```console
conda install -c conda-forge imbalanced-learn
```

7. Install XGBoost Library (https://anaconda.org/conda-forge/xgboost)
```console
conda install -c conda-forge xgboost
```
     
===========================================================================
### Acknowledgments
===========================================================================  
#### References
Johnson, A. E. W., Pollard, T. J., Shen, L., Lehman, L. H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Celi, L. A., & Mark, R. G. (2016). MIMIC-III, a freely accessible critical care database. Scientific data, 3, 160035.

#### MIMIC-III citation

MIMIC-III, a freely accessible critical care database. Johnson AEW, Pollard TJ, Shen L, Lehman L, Feng M, Ghassemi M, Moody B, Szolovits P, Celi LA, and Mark RG. Scientific Data (2016). DOI: 10.1038/sdata.2016.35. Available at: http://www.nature.com/articles/sdata201635

#### Mimic III Data
Pollard, T. J. & Johnson, A. E. W. The MIMIC-III Clinical Database http://dx.doi.org/10.13026/C2XW26 (2016).

#### PhysioNet 
Physiobank, physiotoolkit, and physionet components of a new research resource for complex physiologic signals. Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov P, Mark RG, Mietus JE, Moody GB, Peng C, and Stanley HE. Circulation. 101(23), pe215–e220. 2000.

#### MIMIC Code Repository
Johnson, Alistair EW, David J. Stone, Leo A. Celi, and Tom J. Pollard. “The MIMIC Code Repository: enabling reproducibility in critical care research.” Journal of the American Medical Informatics Association (2017): ocx084.

Mimic III has extensive documentation that I linked below to provide additional information about the data source.

#### Mimic III Data.  
https://mimic.physionet.org/gettingstarted/overview/

#### Mimic III Schema 
https://mit-lcp.github.io/mimic-schema-spy/relationships.html


* The inspiration for expanding the hospital readmission project using unstructured data came from Andrew Long's work on Hosital Readmissions  
    - Introduction to Clinical Natural Language Processing: Predicting Hospital Readmission with Discharge Summaries -  (https://towardsdatascience.com/introduction-to-clinical-natural-language-processing-predicting-hospital-readmission-with-1736d52bc709)
