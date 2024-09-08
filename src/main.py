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
import evaluation as eval

if __name__ == "__main__":

    # Sample data
    true_labels = pd.DataFrame({
        'L1': [1, 0, 1, 1],
        'L2': [0, 1, 1, 0],
        'L3': [0, 0, 1, 1],
        'L4': [1, 1, 0, 1]
    })
    pred_labels = pd.DataFrame({
        'L1': [1, 0, 0, 1],
        'L2': [0, 1, 0, 1],
        'L3': [0, 0, 1, 0],
        'L4': [1, 0, 0, 1]
    })
    pred_scores = pd.DataFrame({
        'L1': [1.0, 0.5, 0.0, 0.6],
        'L2': [0.3, 0.4, 0.1, 0.9],
        'L3': [0.2, 0.6, 0.7, 0.8],
        'L4': [0.9, 0.5, 0.3, 0.1]
    })

    res_bipartition = eval.multilabel_bipartition_measures(true_labels, pred_labels)   
    res_ranking = eval.multilabel_ranking_measures(true_labels, pred_scores)
    res_curves = eval.multilabel_curves_measures(true_labels, pred_scores)
    res_lp = eval.multilabel_label_problem_measures(true_labels, pred_labels)
    
    print(res_bipartition)
    print(res_ranking)
    print(res_curves)
    print(res_lp)