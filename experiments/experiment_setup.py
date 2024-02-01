import os
import inspect
import sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from data_model import DataModelType, SigmaDeltaProcessType
from experiment_information import NumpyEncoder, ExperimentInformation, ExperimentType, ProblemType, DataModelDefinition
import json
import numpy as np
from helpers import *

import logging
logger = logging.getLogger()
# Make the logger log to console
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)



feature_ratios = None
features_x = None
features_theta = None


delete_existing = True
normalize_matrices = False

"""
------------------------------------------------------------------------------------------------------------------------
    Start Here, and define your data_model_definitions
------------------------------------------------------------------------------------------------------------------------
"""
d = 1000

delta_ratio = 4
inverse_delta_ratio = 1/delta_ratio
Sigma_w_content = np.array([1,1])
Sigma_upsilon_content = np.array([1,1])

#### Sweep in Theta Ratio

# data_model_definitions = []
# # data_models_description = "NoNormalisation"
# # data_models_description = "TraceNormalisation"



# sigmax_ratio = 2
# inverse_sigmax_ratio = 1/sigmax_ratio


# data_models_description = f"XSweepUsefulness"



# def get_theta_ratio(rx, ru):
#     # return np.sqrt( np.sqrt( rx / ( 0.5 * ( ru**2 + ru ) ) ) )

#     return np.sqrt( ru  )


# for usefulness_ratio in np.logspace(-1,0.8,10):




#     """ 10 x 1 """

    
    
#     theta_second = usefulness_ratio/sigmax_ratio
#     theta_first = theta_second



#     # round both to 4 digits
#     theta_first = np.round(theta_first,8)
#     theta_second = np.round(theta_second,8)


    

#     data_model_type = DataModelType.KFeaturesModel
#     feature_ratios = np.array([0.5,0.5])
#     features_x = np.array([sigmax_ratio,sigmax_ratio])
#     features_theta = np.array([theta_first,theta_second])
#     data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingIdentity_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
#     data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
#     Sigma_delta_content = np.array([1,1])
#     definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
#     data_model_definitions.append(definition)




#     """ 0.1 x 1 """


#     theta_second = usefulness_ratio/inverse_sigmax_ratio
#     theta_first = theta_second

#     # round both to 4 digits
#     theta_first = np.round(theta_first,8)
#     theta_second = np.round(theta_second,8)


    

#     data_model_type = DataModelType.KFeaturesModel
#     feature_ratios = np.array([0.5,0.5])
#     features_x = np.array([inverse_sigmax_ratio,inverse_sigmax_ratio])
#     features_theta = np.array([theta_first,theta_second])
#     data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingIdentity_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
#     data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
#     Sigma_delta_content = np.array([1,1])
#     definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
#     data_model_definitions.append(definition)


#     """ 1 x 1 """

#     theta_second = usefulness_ratio 
#     theta_first = theta_second

#     # round both to 4 digits
#     theta_first = np.round(theta_first,8)
#     theta_second = np.round(theta_second,8)

#     data_model_type = DataModelType.KFeaturesModel
#     feature_ratios = np.array([0.5,0.5])
#     features_x = np.array([1,1])
#     features_theta = np.array([theta_first,theta_second])
#     data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingIdentity_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
#     data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
#     Sigma_delta_content = np.array([1,1])
#     definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
#     data_model_definitions.append(definition)



"""
--------------------------------- Theta Sweep below ---------------------------------

"""

# data_model_definitions = []

# teacher_ratio = 2
# inverse_teacher_ratio = 1/teacher_ratio

# data_models_description = "ThetaSweepUsefulness"


# for usefulness_ratio in np.logspace(-1,0.8,10):    


#     """ 10 x 1 """

    
    
#     x_first = usefulness_ratio/teacher_ratio
#     x_second = x_first
    



#     # round both to 4 digits
#     x_first = np.round(x_first,8)
#     x_second = np.round(x_second,8)


#     data_model_type = DataModelType.KFeaturesModel
#     feature_ratios = np.array([0.5,0.5])
#     features_x = np.array([x_first, x_second])
#     features_theta = np.array([teacher_ratio,teacher_ratio])
#     data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingIdentity_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
#     data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
#     Sigma_delta_content = np.array([1,1])
#     definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
#     data_model_definitions.append(definition)


#     """ 0.1 x 1 """


#     x_first = usefulness_ratio/inverse_teacher_ratio
#     x_second = x_first



#     # round both to 4 digits
#     x_first = np.round(x_first,8)
#     x_second = np.round(x_second,8)


    

#     data_model_type = DataModelType.KFeaturesModel
#     feature_ratios = np.array([0.5,0.5])
#     features_x = np.array([x_first, x_second])
#     features_theta = np.array([inverse_teacher_ratio,inverse_teacher_ratio])
#     data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingIdentity_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
#     data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
#     Sigma_delta_content = np.array([1,1])
#     definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
#     data_model_definitions.append(definition)




#     """ 1 x 1 """

#     x_first = usefulness_ratio/1
#     x_second = x_first   



#     # round both to 4 digits
#     x_first = np.round(x_first,8)
#     x_second = np.round(x_second,8)

#     data_model_type = DataModelType.KFeaturesModel
#     feature_ratios = np.array([0.5,0.5])
#     features_x = np.array([x_first, x_second])
#     features_theta = np.array([1,1])
#     data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingIdentity_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
#     data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
#     Sigma_delta_content = np.array([1,1])
#     definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
#     data_model_definitions.append(definition)



"""
--------------------------------- Theta Sweep above ---------------------------------
--------------------------------- Tau Sweep below ---------------------------------
"""


# data_model_definitions = []



# data_models_description = f"SweepTau"

# data_model_type = DataModelType.KFeaturesModel
# feature_ratios = np.array([0.5,0.5])
# features_x = np.array([1,1])
# features_theta = np.array([1,1])
# data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingSecondStronger_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
# data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
# Sigma_delta_content = np.array([1,1])
# definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
# data_model_definitions.append(definition)


"""
--------------------------------- Tau Sweep above ---------------------------------
--------------------------------- Attack Epsilon Sweep below ---------------------------------
"""


# data_model_definitions = []



# data_models_description = f"RhoNormalisationSweepEpsilon"

# data_model_type = DataModelType.KFeaturesModel
# feature_ratios = np.array([0.5,0.5])
# features_x = np.array([1,1])
# features_theta = np.array([1,1])
# data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingSecondStronger_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
# data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
# Sigma_delta_content = np.array([1,1])
# definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
# data_model_definitions.append(definition)


"""
--------------------------------- Attack Epsilon Sweep above ---------------------------------
--------------------------------- All Feature Combination Attack Epsilon Sweep below ---------------------------------
"""


data_model_definitions = []



data_models_description = f"FeatureComparisonsTest"

data_model_type = DataModelType.KFeaturesModel
feature_ratios = np.array([0.5,0.5])
features_x = np.array([0.5,0.5])
features_theta = np.array([2,2])
data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingSecondStronger_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
Sigma_delta_content = np.array([1,1])
definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
data_model_definitions.append(definition)

data_model_type = DataModelType.KFeaturesModel
feature_ratios = np.array([0.5,0.5])
features_x = np.array([2,2])
features_theta = np.array([2,2])
data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingSecondStronger_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
Sigma_delta_content = np.array([1,1])
definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
data_model_definitions.append(definition)


data_model_type = DataModelType.KFeaturesModel
feature_ratios = np.array([0.5,0.5])
features_x = np.array([0.5,0.5])
features_theta = np.array([8,8])
data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingSecondStronger_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
Sigma_delta_content = np.array([1,2])
definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
data_model_definitions.append(definition)

data_model_type = DataModelType.KFeaturesModel
feature_ratios = np.array([0.5,0.5])
features_x = np.array([2,2])
features_theta = np.array([1/2,1/2])
data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingSecondStronger_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
Sigma_delta_content = np.array([1,1])
definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
data_model_definitions.append(definition)


"""
--------------------------------- All Feature Combination Attack Epsilon Sweep below above ---------------------------------
--------------------------------- Sweep Defence below ---------------------------------
"""


# data_model_definitions = []



# data_models_description = f"Defence Sweep"

# robustness_ratio = 5
# first_feature = robustness_ratio
# second_feature = 1/robustness_ratio


# data_model_type = DataModelType.KFeaturesModel
# feature_ratios = np.array([0.5,0.5])
# features_x = np.array([first_feature,second_feature])
# features_theta = np.array([1,1])
# Sigma_delta_content = np.array([1,1])
# data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingIdentity_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
# data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"

# definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
# data_model_definitions.append(definition)

# data_model_type = DataModelType.KFeaturesModel
# feature_ratios = np.array([0.5,0.5])
# features_x = np.array([first_feature,second_feature])
# features_theta = np.array([1,1])
# Sigma_delta_content = np.array([4,1])
# data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingFirstStronger_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_2_1_SU_1_1"
# data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta First Protectes"

# definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
# data_model_definitions.append(definition)


# data_model_type = DataModelType.KFeaturesModel
# feature_ratios = np.array([0.5,0.5])
# features_x = np.array([first_feature,second_feature])
# features_theta = np.array([1,1])
# Sigma_delta_content = np.array([1,4])
# data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingSecondStronger_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_2_SU_1_1"
# data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Second Protected"
# definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
# data_model_definitions.append(definition)



"""
--------------------------------- Sweep Defence above ---------------------------------
--------------------------------- Equivalent Loss below ---------------------------------
"""


# data_model_definitions = []



# data_models_description = f"EquivalentSweep"

# data_model_type = DataModelType.KFeaturesModel
# feature_ratios = np.array([0.5,0.5])
# features_x = np.array([1,1])
# features_theta = np.array([1,1])
# data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingSecondStronger_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
# data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
# Sigma_delta_content = np.array([1,1])
# definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
# data_model_definitions.append(definition)


"""
--------------------------------- Equivalent Loss above ---------------------------------
--------------------------------- Optimal Regularisation Search below ---------------------------------
"""


# data_model_definitions = []



# data_models_description = f"ObtainOptimalLambda"
# data_models_description = f"SweepAtOptimalLambda"

# data_model_type = DataModelType.KFeaturesModel
# feature_ratios = np.array([0.5,0.5])
# features_x = np.array([1,1])
# features_theta = np.array([1,1])
# data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingSecondStronger_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
# data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
# Sigma_delta_content = np.array([1,1])
# definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
# data_model_definitions.append(definition)

"""
--------------------------------- Optimal Regularisation Search above ---------------------------------
--------------------------------- PerturbedLogistic below ---------------------------------
"""


# data_model_definitions = []


# data_models_description = f"PerturbedLogistic"

# data_model_type = DataModelType.KFeaturesModel
# feature_ratios = np.array([0.5,0.5])
# features_x = np.array([1,1])
# features_theta = np.array([1,1])
# data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingSecondStronger_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
# data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
# Sigma_delta_content = np.array([1,1])
# definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
# data_model_definitions.append(definition)


"""
--------------------------------- PerturbedLogistic above ---------------------------------
--------------------------------- Power-Law Data below ---------------------------------
"""

# data_models_description = f"PowerLawData"
# data_model_definitions = []

# d = 500

# Sigma_w_content = np.ones(d)
# Sigma_upsilon_content = np.ones(d)

# alpha = 3.5
# data_model_type = DataModelType.KFeaturesModel
# ratio = 1/d
# feature_ratios = ratio * np.ones(d)
# features_x = np.array([10e2*i**-alpha for i in range(1,d+1)])
# features_theta = np.ones(d)
# data_model_name = f"KFeaturesModel_PowerLaw_Coefficient_{alpha}"
# data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
# Sigma_delta_content = np.ones(d)
# definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
# data_model_definitions.append(definition)

# alpha = 2.5
# data_model_type = DataModelType.KFeaturesModel
# ratio = 1/d
# feature_ratios = ratio * np.ones(d)
# features_x = np.array([10e2*i**-alpha for i in range(1,d+1)])
# features_theta = np.ones(d)
# data_model_name = f"KFeaturesModel_PowerLaw_Coefficient_{alpha}"
# data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
# Sigma_delta_content = np.ones(d)
# definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
# data_model_definitions.append(definition)

# alpha = 1.5
# data_model_type = DataModelType.KFeaturesModel
# ratio = 1/d
# feature_ratios = ratio * np.ones(d)
# features_x = np.array([10e2*i**-alpha for i in range(1,d+1)])
# features_theta = np.ones(d)
# data_model_name = f"KFeaturesModel_PowerLaw_Coefficient_{alpha}"
# data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
# Sigma_delta_content = np.ones(d)
# definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
# data_model_definitions.append(definition)

# # alpha = 1.2
# # data_model_type = DataModelType.KFeaturesModel
# # ratio = 1/d
# # feature_ratios = ratio * np.ones(d)
# # features_x = np.array([10e1*i**-alpha for i in range(1,d+1)])
# # features_theta = np.ones(d)
# # data_model_name = f"KFeaturesModel_PowerLaw_Coefficient_{alpha}"
# # data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
# # Sigma_delta_content = np.ones(d)
# # definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
# # data_model_definitions.append(definition)

# alpha = 0.5
# data_model_type = DataModelType.KFeaturesModel
# ratio = 1/d
# feature_ratios = ratio * np.ones(d)
# features_x = np.array([10e2*i**-alpha for i in range(1,d+1)])
# features_theta = np.ones(d)
# data_model_name = f"KFeaturesModel_PowerLaw_Coefficient_{alpha}"
# data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
# Sigma_delta_content = np.ones(d)
# definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
# data_model_definitions.append(definition)



"""
--------------------------------- Power-Law Data above ---------------------------------
--------------------------------- Power-Law Beta Sweep below ---------------------------------
"""


# data_models_description = f"PowerLawBetaSweep"
# data_model_definitions = []

# d = 500

# Sigma_w_content = np.ones(d)
# Sigma_upsilon_content = np.ones(d)


# for alpha in np.linspace(1.01,3,30):

#     # round alpha to 4 digits
#     alpha = np.round(alpha,4)

#     data_model_type = DataModelType.KFeaturesModel
#     ratio = 1/d
#     feature_ratios = ratio * np.ones(d)
#     features_x = np.array([10e2*i**-alpha for i in range(1,d+1)])
#     features_theta = np.ones(d)
#     data_model_name = f"KFeaturesModel_PowerLaw_Coefficient_{alpha}"
#     data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
#     Sigma_delta_content = np.ones(d)
#     definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
#     data_model_definitions.append(definition)


"""
--------------------------------- Power-Law Beta Sweep  above ---------------------------------
--------------------------------- High Alpha Sweep below ---------------------------------
"""


# data_model_definitions = []




# data_models_description = f"HighAlphaSweep"

# data_model_type = DataModelType.KFeaturesModel
# feature_ratios = np.array([0.5,0.5])
# features_x = np.array([1,1])
# features_theta = np.array([1,1])
# data_model_name = f"KFeaturesModel_Vanilla_ProtectingIdentity_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
# data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
# Sigma_delta_content = np.array([1,1])
# definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
# data_model_definitions.append(definition)

"""
--------------------------------- High Alpha Sweep  above ---------------------------------
--------------------------------- Sweep Train-epsilon below ---------------------------------
"""


# data_model_definitions = []




# data_models_description = f"TrainEpsilonSweep"

# data_model_type = DataModelType.KFeaturesModel
# feature_ratios = np.array([0.5,0.5])
# features_x = np.array([1,1])
# features_theta = np.array([1,1])
# data_model_name = f"KFeaturesModel_Vanilla_ProtectingIdentity_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
# data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
# Sigma_delta_content = np.array([1,1])
# definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
# data_model_definitions.append(definition)


# data_model_type = DataModelType.KFeaturesModel
# feature_ratios = np.array([0.5,0.5])
# features_x = np.array([0.1,0.1])
# features_theta = np.array([1,1])
# data_model_name = f"KFeaturesModel_Vanilla_ProtectingIdentity_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
# data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
# Sigma_delta_content = np.array([1,1])
# definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
# data_model_definitions.append(definition)


# data_model_type = DataModelType.KFeaturesModel
# feature_ratios = np.array([0.5,0.5])
# features_x = np.array([10,10])
# features_theta = np.array([1,1])
# data_model_name = f"KFeaturesModel_Vanilla_ProtectingIdentity_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
# data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
# Sigma_delta_content = np.array([1,1])
# definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
# data_model_definitions.append(definition)

"""
--------------------------------- Sweep Train-epsilon  above ---------------------------------
--------------------------------- Optimal Defense below ---------------------------------
"""


# data_model_definitions = []

# data_models_description = f"OptimalDefense"

# d = 1000


# feature_ratios = np.array([0.01,0.99])
# Sigma_w_content = np.array([1,1])

# features_x = np.array([1,1])
# features_theta = np.array([1000,1])



# data_model_type = DataModelType.KFeaturesModel

# Sigma_upsilon_content = np.array([1000,1])

# # Sigma_delta_content = np.array([1,1])
# # data_model_name = f"KFeaturesModel_UniformDefense_TeacherAttacked"
# # data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
# # definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
# # data_model_definitions.append(definition)


# Sigma_delta_content = np.array([1000,1])
# data_model_name = f"KFeaturesModel_TeacherDefense_TeacherAttacked"
# data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
# definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
# data_model_definitions.append(definition)


# Sigma_upsilon_content = np.array([1,1000])

# # Sigma_delta_content = np.array([1,1])
# # data_model_name = f"KFeaturesModel_UniformDefense_OrthogonalAttacked"
# # data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
# # definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
# # data_model_definitions.append(definition)




# Sigma_delta_content = np.array([1,1000])
# data_model_name = f"KFeaturesModel_OrthogonalDefense_OrthogonalAttacked"
# data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
# definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta, attack_equal_defense=True,sigma_delta_process_type=SigmaDeltaProcessType.ComputeTeacherDirection)
# data_model_definitions.append(definition)





"""
--------------------------------- Optimal Defense  above ---------------------------------
--------------------------------- Overlap Scalings below ---------------------------------
"""


# data_models_description = f"OverlapScalings"
# data_model_definitions = []

# data_model_type = DataModelType.KFeaturesModel
# feature_ratios = np.array([0.5,0.5])
# features_x = np.array([1,1])
# # features_theta = np.array([1,1])
# # samples features theta from a normal distribution (0,1)
# # features_theta = np.random.normal(0,1,2)
# data_model_name = f"KFeaturesModel_Vanilla_ProtectingIdentity_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
# data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
# Sigma_delta_content = np.array([5,1])
# definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
# data_model_definitions.append(definition)

"""
--------------------------------- Overlap Scalings  above ---------------------------------
--------------------------------- High Alpha below ---------------------------------
"""


# data_models_description = f"HighAlpha"
# data_model_definitions = []

# data_model_type = DataModelType.KFeaturesModel
# feature_ratios = np.array([0.5,0.5])
# features_x = np.array([1,1])
# features_theta = np.array([1,1])
# data_model_name = f"KFeaturesModel_Vanilla_ProtectingIdentity_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
# data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
# Sigma_delta_content = np.array([1,1])
# definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w_content, Sigma_delta_content, Sigma_upsilon_content, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
# data_model_definitions.append(definition)



"""
--------------------------------- Sweeps in Protection and Data Model ---------------------------------
"""

for definition in data_model_definitions:
    definition.name += "___" + data_models_description


memory = "200G"

# data_models_description = "Vanilla vs KFeatures Identity vs Identity"
# data_models_description = data_model_definitions[0].name

"""
------------------------------------------------------------------------------------------------------------------------
    Next, define your experiment.
------------------------------------------------------------------------------------------------------------------------
"""
# experiment_filename = "HighAlphaSweep/sweep_experiment_9.json"
# experiment_filename = f"AlphaSweepsLam10eM3/{data_models_description}/sweep_experiment.json"
# experiment_filename = f"ThetaSweeps/{data_models_description}/sweep_experiment.json"
# experiment_filename = f"ColorMapSweeps/{data_models_description}/sweep_experiment.json"
experiment_filename = f"ReportSweeps/{data_models_description}/sweep_experiment.json"
# experiment_filename = "sweep_experiment.json"


# extract data_model_names, data_model_descriptions and data_model_types from the data_model_definitions
data_model_names = [data_model_definition.name for data_model_definition in data_model_definitions]
data_model_descriptions = [data_model_definition.description for data_model_definition in data_model_definitions]
data_model_types = [data_model_definition.data_model_type for data_model_definition in data_model_definitions]

# # Create a SweepExperiment

# repetitions
state_evolution_repetitions: int = 1
erm_repetitions: int = 0

# sweeps
alphas: np.ndarray = np.logspace(1,6,10)  # #np.linspace(0.1,10,15)  #
epsilons: np.ndarray =  np.array([0.0,0.1,0.2,0.3]) # np.logspace(-3,0,10) # # # # np.linspace(0,0.6,2) # np.array([0.0,0.2]) # np.array([0,0.1,0.3,0.4,0.5]) 
lambdas: np.ndarray = np.array([1e-3]) #np.logspace(-4,2,15) # np.logspace(-1,2,1) #np.concatenate([-np.logspace(-4,-1,10),np.logspace(-6,-3,2)])  #np.array([-0.0001])
test_against_epsilons: np.ndarray = np.array([0.2])
taus: np.ndarray = np.array([0.05])
# taus: np.ndarray = np.linspace(0.05,1.5,30)

# round the lambdas, epsilons and alphas for 4 digits
alphas = np.round(alphas,4)    
epsilons = np.round(epsilons,4)
lambdas = np.round(lambdas,8)
taus = np.round(taus,4)
test_against_epsilons = np.round(test_against_epsilons,4)



# calibration
ps: np.ndarray = None # np.array([0.6,0.75,0.9])
dp: float = 0.01

# Type, name and problem
# experiment_type: ExperimentType = ExperimentType.OptimalLambdaAdversarialTestError
experiment_type: ExperimentType = ExperimentType.Sweep
# experiment_type: ExperimentType = ExperimentType.OptimalLambda
# experiment_type: ExperimentType = ExperimentType.SweepAtOptimalLambdaAdversarialTestError
experiment_name: str = f"{data_models_description}"
problem_types: list[ProblemType] = [ProblemType.Logistic]

# Fair error
gamma_fair_error: float = 0.0001



experiment = ExperimentInformation(state_evolution_repetitions,erm_repetitions,alphas,epsilons,lambdas,taus,d,experiment_type,ps,dp, data_model_types,data_model_names, data_model_descriptions, test_against_epsilons,problem_types,gamma_fair_error, experiment_name)



# Log the experiment object
logger.info(f"Experiment: {experiment}")

"""
------------------------------------------------------------------------------------------------------------------------
    Now let the experiment create the data_model and save the experiment to a json file. 
------------------------------------------------------------------------------------------------------------------------
"""

experiment.store_data_model_definitions(data_model_definitions)


# make sure that the experiment directory exists
if os.path.dirname(experiment_filename) != "":
    os.makedirs(os.path.dirname(experiment_filename), exist_ok=True)

# use json dump to save the experiment parameters
with open(experiment_filename,"w") as f:
    # use the NumpyEncoder to encode numpy arrays
    # Let's produce some json
    json_string = json.dumps(experiment,cls= NumpyEncoder)
    # now write it
    f.write(json_string)

logger.info(f"Succesfully saved experiment to {experiment_filename}. You can run it even if we did not recreate the data_model!")

if os.path.dirname(experiment_filename) != "":
        
    # Now let's create a batch file to run the experiment
    content = f"""#!/bin/bash
#SBATCH --chdir=/home/ktanner/TPIV---Logistic-Classification/experiments
#SBATCH --job-name=TPIV-Adversarial
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --mem={memory}
#SBATCH --output='{os.path.dirname(experiment_filename)}/out.txt'
#SBATCH --error='{os.path.dirname(experiment_filename)}/error.txt'
#SBATCH --time=24:00:00

module purge
module load gcc openmpi python/3.10.4
source /home/ktanner/venvs/adv/bin/activate

srun --mpi pmi2 python3 ./create_data_model.py '{experiment_filename}'
srun --mpi=pmi2 python3 ./sweep.py '{experiment_filename}'

deactivate
"""

    # save the batch file in the same directory as the experiment
    with open(f"{os.path.dirname(experiment_filename)}/run.sh","w") as f:
        f.write(content)

# # Start the MPI
# import subprocess
# # Run this command
# # mpiexec -n 2 python sweep.py sweep_experiment.json
# subprocess.run(["mpiexec","-n","4","python","sweep.py","sweep_experiment.json"])