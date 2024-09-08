##############################################################################
# Copyright (C) 2024                                                         #
#                                                                            #
# CC BY-NC-SA 4.0                                                            #
#                                                                            #
# Canonical URL https://creativecommons.org/licenses/by-nc-sa/4.0/           #
# Attribution-NonCommercial-ShareAlike 4.0 International CC BY-NC-SA 4.0     #
#                                                                            #
# Prof. Elaine Cecilia Gatto | Prof. Ricardo Cerri | Prof. Mauri Ferrandin   #
#                                                                            #
# Federal University of São Carlos - UFSCar - https://www2.ufscar.br         #
# Campus São Carlos - Computer Department - DC - https://site.dc.ufscar.br   #
# Post Graduate Program in Computer Science - PPGCC                          # 
# http://ppgcc.dc.ufscar.br - Bioinformatics and Machine Learning Group      #
# BIOMAL - http://www.biomal.ufscar.br                                       #
#                                                                            #
# You are free to:                                                           #
#     Share — copy and redistribute the material in any medium or format     #
#     Adapt — remix, transform, and build upon the material                  #
#     The licensor cannot revoke these freedoms as long as you follow the    #
#       license terms.                                                       #
#                                                                            #
# Under the following terms:                                                 #
#   Attribution — You must give appropriate credit , provide a link to the   #
#     license, and indicate if changes were made . You may do so in any      #
#     reasonable manner, but not in any way that suggests the licensor       #
#     endorses you or your use.                                              #
#   NonCommercial — You may not use the material for commercial purposes     #
#   ShareAlike — If you remix, transform, or build upon the material, you    #
#     must distribute your contributions under the same license as the       #
#     original.                                                              #
#   No additional restrictions — You may not apply legal terms or            #
#     technological measures that legally restrict others from doing         #
#     anything the license permits.                                          #
#                                                                            #
##############################################################################





########################################################################
#                                                                      #
########################################################################
import sys
import platform
import os

system = platform.system()
if system == 'Windows':
    user_profile = os.environ['USERPROFILE']
    FolderRoot = os.path.join(user_profile, 'Documents', 'MultiLabelEvaluationMetrics', 'src')
elif system in ['Linux', 'Darwin']:  # 'Darwin' is the system name for macOS
    FolderRoot = os.path.expanduser('~/MultiLabelEvaluationMetrics/src')
else:
    raise Exception('Unsupported operating system')

os.chdir(FolderRoot)
current_directory = os.getcwd()
sys.path.append('..')


import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, hamming_loss, zero_one_loss,
    average_precision_score, f1_score, precision_score,
    recall_score, jaccard_score, roc_auc_score, precision_recall_curve,
    precision_recall_fscore_support, roc_curve, auc, coverage_error, 
    label_ranking_loss
)

import confusion_matrix as cm
import measures as ms



########################################################################
#                                                                      #
########################################################################
def multilabel_label_problem_measures(true_labels: pd.DataFrame, pred_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates measures for label prediction problems in multi-label classification.

    Parameters:
    ----------
    true_labels (pd.DataFrame): The DataFrame containing the true binary labels (0 or 1) for each instance.
    pred_labels (pd.DataFrame): The DataFrame containing the predicted binary labels (0 or 1) for each instance.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing all the calculated metrics.

    Metrics Calculated:
    -------------------
    - Constant Label Problem (CLP)
    - Wrong Label Problem (WLP)
    - Missing Label Problem (MLP)

    Interpretation:
    ----------
    1. **Wrong Label Problem (WLP)**
        Definition: Measures the number of labels that are predicted but should not be. The ideal value is zero.        
        - **Low WLP**: Indicates fewer incorrect predictions of labels.
        - **High WLP**: Indicates that the classifier often predicts incorrect labels.
        - **Reference**: Rivolli, A., Soares, C., & Carvalho, A. C. P. de L. F. de. (2018). Enhancing 
        multilabel classification for food truck recommendation. Expert Systems. Wiley-Blackwell. 
        DOI: 10.1111/exsy.12304

    2. **Missing Label Problem (MLP)**
        Definition: Measures the proportion of labels that should have been predicted but were not. The ideal value is zero.        
        - **Low MLP**: Indicates that most of the relevant labels are predicted.
        - **High MLP**: Indicates that many relevant labels are missing in the predictions.
        - **Reference**: Rivolli, A., Soares, C., & Carvalho, A. C. P. de L. F. de. (2018). Enhancing 
        multilabel classification for food truck recommendation. Expert Systems. Wiley-Blackwell. 
        DOI: 10.1111/exsy.12304

    3. **Constant Label Problem (CLP)**
        Definition: Measures the occurrence where the same label is predicted for all instances. The ideal value is zero.        
        - **Low CLP**: Indicates that predictions vary and are more closely aligned with true labels.
        - **High CLP**: Indicates that the classifier predicts the same label for all instances.
        - **Reference**: Rivolli, A., Soares, C., & Carvalho, A. C. P. de L. F. de. (2018). Enhancing 
        multilabel classification for food truck recommendation. Expert Systems. Wiley-Blackwell. 
        DOI: 10.1111/exsy.12304
  

    Example Usage:
    --------------
    >>> result_df = multilabel_label_problem_measures(true_labels, pred_labels)
    >>> print(result_df)
    """

    matrix_confusion = cm.mlem_confusion_matrix(true_labels, pred_labels)

    clp = ms.mlem_clp(matrix_confusion)
    mlp = ms.mlem_mlp(matrix_confusion)
    wlp = ms.mlem_wlp(matrix_confusion)

    # Store all metrics in a dictionary
    metrics_dict = {    
        'clp': clp,
        'mlp': mlp,
        'wlp': wlp
    }

    # Convert dictionary to DataFrame
    # metrics_df = pd.DataFrame([metrics_dict])

    # Converter o dicionário em um DataFrame com colunas "Measure" e "Value"
    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=['Measure', 'Value'])

    return metrics_df



########################################################################
#                                                                      #
########################################################################
def multilabel_bipartition_measures(true_labels: pd.DataFrame, pred_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates various evaluation metrics for multi-label classification.

    Parameters:
    ----------
    true_labels (pd.DataFrame): The DataFrame containing the true binary labels (0 or 1) for each instance.
    pred_labels (pd.DataFrame): The DataFrame containing the predicted binary labels (0 or 1) for each instance.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing all the calculated metrics.

    Metrics Calculated:
    -------------------
    - Accuracy
    - Hamming Loss
    - Zero-One Loss
    - F1 Score (macro, micro, weighted, samples)
    - Precision (macro, micro, weighted, samples)
    - Recall (macro, micro, weighted, samples)
    - Precision Recall F1 Support (macro, micro, weighted, samples)
    - Jaccard Score (macro, micro, weighted, samples)

    Interpretation:
    ----------------
    1. **Accuracy**
        Definition: The proportion of correctly predicted labels (both positive and negative) over 
        the total number of labels.
        - **High Accuracy**: Indicates that the classifier correctly predicted a high proportion of labels.
        - **Low Accuracy**: Indicates that the classifier made many incorrect predictions.
        - **Reference**: [Wikipedia: Accuracy and Precision](https://en.wikipedia.org/wiki/Accuracy_and_precision)

    2. **Hamming Loss**
        Definition: The fraction of labels that are incorrectly predicted, either due to false
        positives or false negatives, normalized by the total number of labels.    
        - **Low Hamming Loss**: Indicates fewer incorrect predictions.
        - **High Hamming Loss**: Indicates many incorrect predictions.
        - **Reference**: [Hamming Loss on scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html)

    3. **Subset Accuracy**
        Definition: The proportion of instances for which the classifier predicted all the labels 
        exactly right (i.e., the predicted label set matches the true label set exactly).
        - **High Subset Accuracy**: Indicates that the classifier correctly predicted all labels for 
          many instances.
        - **Low Subset Accuracy**: Indicates that the classifier often missed some labels or included 
          incorrect labels.
        - **Reference**: [Subset Accuracy on scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)

    4. **Zero-One Loss**
        Definition: The fraction of instances where the classifier’s prediction does not match the 
        true label set (i.e., the prediction is not an exact match).    
        - **Low Zero-One Loss**: Indicates that the classifier makes fewer predictions that do not match 
          the true labels exactly.
        - **High Zero-One Loss**: Indicates that the classifier often makes incorrect predictions.
        - **Reference**: [Zero-One Loss on scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.zero_one_loss.html)

    5. **Precision (Macro)**
        Definition: The average precision score calculated for each label independently and then 
        averaged, treating all labels equally.    
        - **High Macro Precision**: Indicates good performance across all labels individually.
        - **Low Macro Precision**: Indicates poor performance on some labels.
        - **Reference**: [Precision Score on scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)

    6. **Precision (Micro)**
        Definition: The total number of true positives divided by the total number of true positives 
        and false positives, aggregated across all labels.    
        - **High Micro Precision**: Indicates good overall performance when considering all labels collectively.
        - **Low Micro Precision**: Indicates many false positives relative to true positives.
        - **Reference**: [Precision Score on scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)

    7. **Precision (Weighted)**
        Definition: The precision score calculated for each label, weighted by the number of true 
        instances for each label, and then averaged.
        - **High Weighted Precision**: Indicates good performance when accounting for the number of 
          instances for each label.
        - **Low Weighted Precision**: Indicates varying performance across labels.
        - **Reference**: [Precision Score on scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)

    8. **Precision (Samples)**
        Definition: The precision score computed for each instance individually, then averaged.    
        - **High Sample Precision**: Indicates good performance on average across different instances.
        - **Low Sample Precision**: Indicates that the classifier often makes incorrect predictions for some instances.
        - **Reference**: [Precision Score on scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)

    9. **Recall (Macro)**
        Definition: The average recall score calculated for each label independently and then 
        averaged, treating all labels equally.    
        - **High Macro Recall**: Indicates good identification of relevant labels across all labels individually.
        - **Low Macro Recall**: Indicates that the classifier misses many relevant labels.
        - **Reference**: [Recall Score on scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)

    10. **Recall (Micro)**
        Definition: The total number of true positives divided by the total number of true positives 
        and false negatives, aggregated across all labels.    
        - **High Micro Recall**: Indicates good overall identification of relevant labels.
        - **Low Micro Recall**: Indicates that the classifier misses many relevant labels.
        - **Reference**: [Recall Score on scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)

    11. **Recall (Weighted)**
        Definition: The recall score calculated for each label, weighted by the number of true 
        instances for each label, and then averaged.    
        - **High Weighted Recall**: Indicates good performance when considering the number of instances for each label.
        - **Low Weighted Recall**: Indicates varying performance in identifying relevant labels.
        - **Reference**: [Recall Score on scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)

    12. **Recall (Samples)**
        Definition: The recall score computed for each instance individually, then averaged.    
        - **High Sample Recall**: Indicates good identification of relevant labels on average across instances.
        - **Low Sample Recall**: Indicates that the classifier misses many relevant labels for some instances.
        - **Reference**: [Recall Score on scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)

    13. **F1 Score (Macro)**
        Definition: The average F1 score calculated for each label independently and then averaged, 
        treating all labels equally.    
        - **High Macro F1**: Indicates a good balance between precision and recall across all labels.
        - **Low Macro F1**: Indicates poor balance between precision and recall.
        - **Reference**: [F1 Score on scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

    14. **F1 Score (Micro)**
        Definition: The total number of true positives divided by the total number of true positives, 
        false positives, and false negatives, aggregated across all labels.
        - **High Micro F1**: Indicates good overall balance between precision and recall.
        - **Low Micro F1**: Indicates poor overall balance between precision and recall.
        - **Reference**: [F1 Score on scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

    15. **F1 Score (Weighted)**
        Definition: The F1 score calculated for each label, weighted by the number of true instances
          for each label, and then averaged.    
        - **High Weighted F1**: Indicates good performance considering the number of instances for each label.
        - **Low Weighted F1**: Indicates varying performance across labels.
        - **Reference**: [F1 Score on scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

    16. **F1 Score (Samples)**
        Definition: The F1 score computed for each instance individually, then averaged.
        - **High Sample F1**: Indicates good balance between precision and recall for each instance.
        - **Low Sample F1**: Indicates that the balance between precision and recall varies significantly across instances.
        - **Reference**: [F1 Score on scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

    17. **Jaccard Score (Macro)**
        Definition: The average Jaccard score computed for each label independently and then 
        averaged, treating all labels equally.
        - **High Macro Jaccard Score**: Indicates good performance across all labels individually.
        - **Low Macro Jaccard Score**: Indicates poor performance on some labels.
        - **Reference**: [Jaccard Score on scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html)

    18. **Jaccard Score (Micro)**
        Definition: The total number of true positives divided by the total number of true positives,
        false positives, and false negatives, aggregated across all labels.
        - **High Micro Jaccard Score**: Indicates good overall performance in terms of similarity and diversity.
        - **Low Micro Jaccard Score**: Indicates poor performance in capturing similarities and differences across labels.
        - **Reference**: [Jaccard Score on scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html)

    19. **Jaccard Score (Weighted)**
        Definition: The Jaccard score calculated for each label, weighted by the number of true 
        instances for each label, and then averaged.
        - **High Weighted Jaccard Score**: Indicates good performance considering the number of instances for each label.
        - **Low Weighted Jaccard Score**: Indicates varying performance across labels.
        - **Reference**: [Jaccard Score on scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html)

    20. **Jaccard Score (Samples)**
        Definition: The Jaccard score computed for each instance individually, then averaged.
        - **High Sample Jaccard Score**: Indicates good performance on average for each instance.
        - **Low Sample Jaccard Score**: Indicates varying performance across instances.
        - **Reference**: [Jaccard Score on scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html)

    Example Usage:
    --------------
    >>> result_df = multilabel_bipartition_measures(true_labels, pred_labels)
    >>> print(result_df)
    """

    # Basic metrics
    accuracy_mlem = ms.mlem_accuracy(true_labels, pred_labels)
    hamming_l = hamming_loss(np.array(true_labels), np.array(pred_labels))    
    zol = zero_one_loss(np.array(true_labels), np.array(pred_labels))    
    sa = ms.mlem_subset_accuracy(true_labels, pred_labels)

    # Precision Scores
    precision_macro = precision_score(true_labels, pred_labels, average='macro', zero_division='warn')    
    precision_micro = precision_score(true_labels, pred_labels, average='micro', zero_division='warn')
    precision_weighted = precision_score(true_labels, pred_labels, average='weighted', zero_division='warn')
    precision_samples = precision_score(true_labels, pred_labels, average='samples', zero_division='warn')
    precision_none = precision_score(true_labels, pred_labels, average=None, zero_division='warn')
    
    # Recall Scores
    recall_macro = recall_score(true_labels, pred_labels, average='macro', zero_division='warn')  
    recall_micro = recall_score(true_labels, pred_labels, average='micro', zero_division='warn')
    recall_weighted = recall_score(true_labels, pred_labels, average='weighted', zero_division='warn')
    recall_samples = recall_score(true_labels, pred_labels, average='samples', zero_division='warn')
    recall_none = recall_score(true_labels, pred_labels, average=None, zero_division='warn')
       
    # F1 Scores
    f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division='warn')
    f1_micro = f1_score(true_labels, pred_labels, average='micro', zero_division='warn')
    f1_weighted = f1_score(true_labels, pred_labels, average='weighted', zero_division='warn')
    f1_samples = f1_score(true_labels, pred_labels, average='samples', zero_division='warn')
    f1_none = f1_score(true_labels, pred_labels, average=None, zero_division='warn')

    # Jaccard Scores
    jaccard_macro = jaccard_score(true_labels, pred_labels, average='macro', zero_division="warn")
    jaccard_micro = jaccard_score(true_labels, pred_labels, average='micro', zero_division="warn")
    jaccard_weighted = jaccard_score(true_labels, pred_labels, average='weighted', zero_division="warn")
    jaccard_samples = jaccard_score(true_labels, pred_labels, average='samples', zero_division="warn")    
    jaccard_none = jaccard_score(true_labels, pred_labels, average=None, zero_division="warn")    

    # Precision, Recall, F1, and Support Scores
    rpf_macro = precision_recall_fscore_support(true_labels, pred_labels, average='macro', zero_division="warn")
    rpf_micro = precision_recall_fscore_support(true_labels, pred_labels, average='micro', zero_division="warn")
    rpf_weighted = precision_recall_fscore_support(true_labels, pred_labels, average='weighted', zero_division="warn")
    rpf_samples = precision_recall_fscore_support(true_labels, pred_labels, average='samples', zero_division="warn")    
    rpf_none = precision_recall_fscore_support(true_labels, pred_labels, average=None, zero_division="warn")    

    # Store all metrics in a dictionary
    metrics_dict = {
        'accuracy': accuracy_mlem,        
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'f1_samples': f1_samples, 
        'hamming_loss': hamming_l,              
        'jaccard_macro': jaccard_macro,
        'jaccard_micro': jaccard_micro,
        'jaccard_weighted': jaccard_weighted,
        'jaccard_samples': jaccard_samples,
        'precision_macro': precision_macro,
        'precision_micro': precision_micro,
        'precision_weighted': precision_weighted,
        'precision_samples': precision_samples,        
        'recall_macro': recall_macro,
        'recall_micro': recall_micro,
        'recall_weighted': recall_weighted,
        'recall_samples': recall_samples,  
        #'precision_recall_fscore_support_macro': rpf_macro,
        #'precision_recall_fscore_support_micro': rpf_micro,
        #'precision_recall_fscore_support_weighted': rpf_weighted,
        #'precision_recall_fscore_support_samples': rpf_samples,
        'zero_one_loss': zol     
    }

    # Convert dictionary to DataFrame
    # metrics_df = pd.DataFrame([metrics_dict])

    # Converter o dicionário em um DataFrame com colunas "Measure" e "Value"
    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=['Measure', 'Value'])

    return metrics_df




########################################################################
#                                                                      #
########################################################################
def multilabel_curves_measures(true_labels: pd.DataFrame, pred_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates various evaluation metrics related to ranking curves for multi-label classification.

    Parameters:
    ----------
    true_labels (pd.DataFrame): The DataFrame containing the true binary labels (0 or 1) for each instance.
    pred_scores (pd.DataFrame): The DataFrame containing the predicted probabilities for each label.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the computed curve-based metrics.

    Metrics Computed:
    ------------------
    - Average Precision (AP) Score (Macro, Micro, Weighted, Samples)
    - ROC AUC Score (Macro, Micro, Weighted, Samples)

    Interpretation:
    ----------------
    1. **Average Precision (AP) Score**
        Definition: Measures the quality of the ranking of predicted probabilities. It summarizes the 
        precision-recall curve by calculating the average precision over all instances.
        - **AP Macro**: The average precision score calculated for each label independently and then 
          averaged, treating all labels equally.
          - High AP Macro: Indicates good performance across all labels, regardless of class imbalance.
        - **AP Micro**: The average precision score calculated by aggregating the contributions of all labels 
          to compute the average precision.
          - High AP Micro: Indicates good overall performance when considering the aggregate precision.
        - **AP Weighted**: The average precision score calculated for each label, weighted by the number of 
          true instances for each label, and then averaged.
          - High AP Weighted: Indicates good performance when considering the number of instances for each label.
        - **AP Samples**: The average precision score computed for each instance individually and then averaged.
          - High AP Samples: Indicates good performance on average across different instances.

    2. **ROC AUC Score**
        Definition: Measures the area under the Receiver Operating Characteristic (ROC) curve, summarizing the 
        trade-off between true positive rate and false positive rate.
        - **ROC AUC Macro**: The ROC AUC score calculated for each label independently and then averaged, 
          treating all labels equally.
          - High ROC AUC Macro: Indicates good performance across all labels, regardless of class imbalance.
        - **ROC AUC Micro**: The ROC AUC score calculated by aggregating the contributions of all labels to 
          compute the average ROC AUC.
          - High ROC AUC Micro: Indicates good overall performance when considering the aggregate true positive 
            rate and false positive rate.
        - **ROC AUC Weighted**: The ROC AUC score calculated for each label, weighted by the number of true 
          instances for each label, and then averaged.
          - High ROC AUC Weighted: Indicates good performance when considering the number of instances for each label.
        - **ROC AUC Samples**: The ROC AUC score computed for each instance individually and then averaged.
          - High ROC AUC Samples: Indicates good performance on average across different instances.

    Example Usage:
    --------------
    >>> result_df = multilabel_curves_measures(true_labels, pred_scores)
    >>> print(result_df)
    """

    # Average Precision Scores
    average_precision_macro = average_precision_score(true_labels, pred_scores, average='macro')
    average_precision_micro = average_precision_score(true_labels, pred_scores, average='micro')
    average_precision_weighted = average_precision_score(true_labels, pred_scores, average='weighted')
    average_precision_samples = average_precision_score(true_labels, pred_scores, average='samples')    
    
    # ROC AUC Scores
    roc_auc_macro = roc_auc_score(true_labels, pred_scores, average='macro')
    roc_auc_micro = roc_auc_score(true_labels, pred_scores, average='micro')
    roc_auc_weighted = roc_auc_score(true_labels, pred_scores, average='weighted')
    roc_auc_samples = roc_auc_score(true_labels, pred_scores, average='samples')      

    # Store all metrics in a dictionary
    metrics_dict = {
        'auprc_macro': average_precision_macro,
        'auprc_micro': average_precision_micro,
        'auprc_weighted': average_precision_weighted,
        'auprc_samples': average_precision_samples,
        'roc_auc_macro': roc_auc_macro,
        'roc_auc_micro': roc_auc_micro,
        'roc_auc_weighted': roc_auc_weighted,
        'roc_auc_samples': roc_auc_samples
    }

    # Convert dictionary to DataFrame
    # metrics_df = pd.DataFrame([metrics_dict])

    # Converter o dicionário em um DataFrame com colunas "Measure" e "Value"
    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=['Measure', 'Value'])

    return metrics_df


    

########################################################################
#                                                                      #
########################################################################
import pandas as pd
import numpy as np
from sklearn.metrics import label_ranking_loss, coverage_error

def mlem_ranking(pred_scores):
    """
    Compute the ranking scores based on prediction scores. This is a placeholder function.

    Parameters:
    ----------
    pred_scores : pd.DataFrame
        A DataFrame where each row represents a sample and each column represents a label.
        The values are prediction scores for each label.

    Returns:
    -------
    pd.DataFrame
        A DataFrame where each row represents a sample and each column represents a label.
        The values indicate the ranking score for each label, where lower values mean higher rank.
    """
    # Placeholder function: replace with actual ranking function
    return pred_scores.rank(axis=1, method='min').astype(int)

def mlem_one_error(true_labels: pd.DataFrame, pred_scores: pd.DataFrame) -> float:
    """
    Compute the One Error metric for multi-label classification.

    The One Error metric measures the proportion of instances where the highest-ranked label is not in the set of true labels.

    Parameters:
    ----------
    true_labels : pd.DataFrame
        A DataFrame where each row represents a sample and each column represents a label.
        The values are binary (0 or 1) indicating the presence or absence of the label.
    pred_scores : pd.DataFrame
        A DataFrame where each row represents a sample and each column represents a label.
        The values are prediction scores for each label.

    Returns:
    -------
    float
        The One Error metric value.

    Example:
    -------
    >>> true_labels = pd.DataFrame([[1, 0, 0], [0, 1, 1], [1, 1, 0]], columns=['A', 'B', 'C'])
    >>> pred_scores = pd.DataFrame([[0.2, 0.5, 0.3], [0.4, 0.2, 0.6], [0.7, 0.1, 0.2]], columns=['A', 'B', 'C'])
    >>> mlem_one_error(true_labels, pred_scores)
    0.6666666666666666

    References:
    ----------
    Schapire, R. E., & Singer, Y. (2000). BoosTexter: A boosting-based system for text categorization. 
    Machine Learning, 39(2), 135-168.
    """
    true_labels = true_labels.to_numpy()
    pred_scores = pred_scores.to_numpy()
    
    # Obtain ranking from prediction scores
    ranking = mlem_ranking(pd.DataFrame(pred_scores, columns=pred_scores.columns))
    
    # Determine the highest-ranked label for each sample
    predicted_labels = np.argmin(ranking.to_numpy(), axis=1)
    
    # Compute the One Error metric
    errors = np.array([1 if true_labels[i, predicted_labels[i]] == 0 else 0 for i in range(true_labels.shape[0])])
    oe = np.mean(errors)
    
    return oe

def multilabel_ranking_measures(true_labels: pd.DataFrame, pred_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates various ranking-based evaluation metrics for multi-label classification.

    Parameters:
    ----------
    true_labels (pd.DataFrame): The DataFrame containing the true binary labels for each instance.
    pred_scores (pd.DataFrame): The DataFrame containing the predicted scores for each label.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the computed ranking-based metrics.

    Metrics Computed:
    ------------------
    - Average Precision
    - Coverage Error
    - Is Error
    - Margin Loss
    - Ranking Error
    - Ranking Loss
    - One Error

    Interpretation:
    ----------------
    1. **Average Precision**
        Definition: Measures the quality of the ranking of predicted labels. It is the average of the 
        precision scores calculated at each position in the ranked list of predictions, weighted by 
        the number of relevant items found.
        - A value of 1.0 indicates perfect ranking where all relevant labels are ranked above all 
          irrelevant labels for every instance.
        - Lower values indicate that the model is not effectively ranking all relevant labels before 
          irrelevant ones.

    2. **Coverage Error**
        Definition: Measures the average number of labels that need to be checked before finding all 
        relevant labels for each instance.
        - A value of 3.5 indicates that, on average, you need to check 3.5 labels to find all relevant 
          labels.
        - Lower values are preferable as they suggest that fewer labels need to be checked to find all 
          relevant ones, indicating better model performance.

    3. **Is Error**
        Definition: Indicates whether there is any discrepancy between the predicted ranking and the true 
        ranking. 
        - A value of 1.0 suggests that there is an error in the ranking, meaning that the predicted 
          ranking does not match the true ranking exactly.
        - A value of 0.0 indicates that the predicted ranking matches the true ranking exactly.

    4. **Margin Loss**
        Definition: Measures the average number of positions by which positive labels are ranked below 
        negative labels. 
        - A Margin Loss value of 1.25 indicates that, on average, positive labels are ranked 1.25 
          positions below negative labels.
        - Lower values are preferable as they suggest that positive labels are ranked closer to the top 
          compared to negative labels.

    5. **Ranking Error**
        Definition: Calculates the sum of squared differences between the predicted and true rankings. 
        - A value of 9.5 indicates the total magnitude of the ranking errors.
        - Lower values are better, indicating that the predicted ranking is closer to the true ranking.

    6. **Ranking Loss**
        Definition: Measures the fraction of label pairs where the ranking is incorrect. 
        - A value of approximately 0.67 indicates that about 67% of label pairs are ranked incorrectly.
        - Lower values are preferred, indicating that the majority of label pairs are ranked correctly.

    7. **One Error**
        Definition: Measures the proportion of instances where the highest-ranked label is not in the set 
        of true labels.
        - A value of 0.0 indicates that for every instance, the highest-ranked label is always a true label.
        - A value of 1.0 indicates that for every instance, the highest-ranked label is never a true label.
        - Lower values are better, indicating that the model is effective at ranking at least one relevant
        label as the highest-ranked label for each instance.

    References:
    ----------
    - The metrics used are commonly referenced in multi-label ranking evaluation literature and libraries.
    - For detailed explanations, see the respective methods in the `ms` (multi-label metrics) library 
    documentation and scikit-learn documentation for `label_ranking_loss` and `coverage_error`.

    Examples:
    ----------
    >>> true_labels = pd.DataFrame([[1, 0, 0], [0, 1, 1], [1, 1, 0]], columns=['A', 'B', 'C'])
    >>> pred_scores = pd.DataFrame([[0.2, 0.5, 0.3], [0.4, 0.2, 0.6], [0.7, 0.1, 0.2]], columns=['A', 'B', 'C'])
    >>> result_df = multilabel_ranking_measures(true_labels, pred_scores)
    >>> print(result_df)
    """
    
    # Compute the various ranking metrics
    average_precision = ms.mlem_average_precision(true_labels, pred_scores)
    precision_atk = ms.mlem_precision_at_k(true_labels, pred_scores)
    coverage = coverage_error(true_labels, pred_scores)
    iserror = ms.mlem_is_error(true_labels, pred_scores)
    margin_loss = ms.mlem_margin_loss(true_labels, pred_scores)       
    ranking_error = ms.mlem_ranking_error(true_labels, pred_scores)       
    ranking_loss = label_ranking_loss(true_labels, pred_scores)       
    one_error = mlem_one_error(true_labels, pred_scores)

    # Store all metrics in a dictionary
    metrics_dict = {    
        'average_precision': average_precision,
        'coverage': coverage,
        'is_error': iserror,
        'margin_loss': margin_loss,
        'precision_atk': precision_atk,
        'ranking_error': ranking_error,
        'ranking_loss': ranking_loss,    
        'one_error': one_error
    }

    # Convert dictionary to DataFrame
    metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=['Measure', 'Value'])

    return metrics_df
