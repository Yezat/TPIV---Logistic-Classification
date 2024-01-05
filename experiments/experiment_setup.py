import os
import inspect
import sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from data_model import DataModelType
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
normalize_matrices = True

"""
------------------------------------------------------------------------------------------------------------------------
    Start Here, and define your data_model_definitions
------------------------------------------------------------------------------------------------------------------------
"""
d = 1000

data_model_type = DataModelType.VanillaGaussian
data_model_name = "VanillaGaussian"
data_model_description = "A Data-Model with Identity Gaussians for all the covariances."
Sigma_w = np.eye(d)
Sigma_delta = np.eye(d)
Sigma_upsilon = np.eye(d) 
Sigma_upsilon = Sigma_upsilon * d / np.trace(Sigma_upsilon)
vanilla_gaussian_data_model_definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type)


## theta first half strong
data_model_type = DataModelType.KFeaturesModel
data_model_name = "VanillaGaussianThetaFirst"
data_model_description = "A Data-Model with Identity Gaussians for all the covariances. Except Theta, which is 10*Identity for the first half."
feature_ratios = np.array([0.5, 0.5])
features_x = np.array([1,1])
features_theta = np.array([10,1])
Sigma_w = np.eye(d)
Sigma_delta = np.eye(d)
Sigma_upsilon = np.eye(d)
Sigma_upsilon = Sigma_upsilon * d / np.trace(Sigma_upsilon)
vanilla_gaussian_theta_first = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)


########## Increasing the variance for all features ##########

## times 1000
# data_model_type = DataModelType.KFeaturesModel
# data_model_name = "VanillaGaussianTimes1000"
# data_model_description = "A Data-Model with Identity Gaussians times 1000 for all the covariances."
# feature_ratios = np.array([1.0])
# features_x = np.array([1000])
# features_theta = np.array([1,1])
# Sigma_w = np.eye(d)
# Sigma_delta = np.eye(d)
# Sigma_upsilon = np.eye(d)
# vanilla_gaussian_times_1000_data_model_definition

# ## times 100
# data_model_type = DataModelType.KFeaturesModel
# data_model_name = "VanillaGaussianTimes100"
# data_model_description = "A Data-Model with Identity Gaussians times 100 for all the covariances."
# feature_ratios = np.array([1.0])
# features_x = np.array([100])
# features_theta = np.array([1,1])
# Sigma_w = np.eye(d)
# Sigma_delta = np.eye(d)
# Sigma_upsilon = np.eye(d)

## times 10
data_model_type = DataModelType.KFeaturesModel
data_model_name = "VanillaGaussianTimes10"
data_model_description = "A Data-Model with Identity Gaussians times 10 for all the covariances."
feature_ratios = np.array([1.0])
features_x = np.array([10])
features_theta = np.array([1,1])
Sigma_w = np.eye(d)
Sigma_delta = np.eye(d)
Sigma_upsilon = np.eye(d)
Sigma_upsilon = Sigma_upsilon * d / np.trace(Sigma_upsilon)
vanilla_gaussian_times_10_data_model_definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)

## times 10
data_model_type = DataModelType.KFeaturesModel
data_model_name = "VanillaGaussianTimes10AttackingFirstHalf"
data_model_description = "A Data-Model with Identity Gaussians times 10 for all the covariances. We attack the first half"
feature_ratios = np.array([1.0])
features_x = np.array([10])
features_theta = np.array([1,1])
Sigma_w = np.eye(d)
Sigma_delta = np.eye(d)
Sigma_upsilon = np.zeros((d,d))
Sigma_upsilon[0:int(d/2),0:int(d/2)] = 5*np.eye(int(d/2))
Sigma_upsilon = Sigma_upsilon * d / np.trace(Sigma_upsilon)
vanilla_gaussian_times_10_attacking_first_data_model_definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)


## times 10
data_model_type = DataModelType.KFeaturesModel
data_model_name = "VanillaGaussianTimes10AttackingSecondHalf"
data_model_description = "A Data-Model with Identity Gaussians times 10 for all the covariances."
feature_ratios = np.array([1.0])
features_x = np.array([10])
features_theta = np.array([1,1])
Sigma_w = np.eye(d)
Sigma_delta = np.eye(d)
Sigma_upsilon = np.zeros((d,d))
Sigma_upsilon[int(d/2):d,int(d/2):d] = 5*np.eye(int(d/2))
Sigma_upsilon = Sigma_upsilon * d / np.trace(Sigma_upsilon)
vanilla_gaussian_times_10_attacking_second_data_model_definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)



########## Attacking the first half of the features ##########

data_model_type = DataModelType.VanillaGaussian
data_model_name = "VanillaGaussianAttackingFirstHalf"
data_model_description = "A Data-Model with Identity Gaussians for all the covariances. But we attack the first half of the features."
Sigma_w = np.eye(d)
Sigma_delta = np.eye(d)
Sigma_upsilon = np.zeros((d,d))
Sigma_upsilon[0:int(d/2),0:int(d/2)] = 5*np.eye(int(d/2))
Sigma_upsilon = Sigma_upsilon * d / np.trace(Sigma_upsilon)
vanilla_gaussian_attack_first_half_data_model_definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type)

########## Attacking the second half of the features ##########

data_model_type = DataModelType.VanillaGaussian
data_model_name = "VanillaGaussianAttackingSecondHalf"
data_model_description = "A Data-Model with Identity Gaussians for all the covariances. But we attack the second half of the features."
Sigma_w = np.eye(d)
Sigma_delta = np.eye(d)
Sigma_upsilon = np.zeros((d,d))
Sigma_upsilon[int(d/2):d,int(d/2):d] = 5*np.eye(int(d/2))
Sigma_upsilon = Sigma_upsilon * d / np.trace(Sigma_upsilon)
vanilla_gaussian_attack_second_half_data_model_definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type)


"""
-------------------
"""

# data_model_type = DataModelType.MarginGaussian
# data_model_name = "DirectedMarginGaussian"
# data_model_description = "A Data-Model with Gaussian Mixtures with a certain margin towards the e1 teacher."
# Sigma_w = np.eye(d)
# Sigma_delta = np.zeros((d,d))
# Sigma_delta[0,0] = 1


"""
-------------------
"""

# data_model_type = DataModelType.SourceCapacity
# data_model_name = "SourceCapacity_IDStudentPrior"
# data_model_description = "Source Capacity data model with Identity Gaussian Prior and a teacher with alpha = 1.2, r = 0.3"
# Sigma_w = np.eye(d)
# Sigma_delta = np.eye(d)

"""
-------------------
"""

# data_model_type = DataModelType.SourceCapacity
# data_model_name = "SourceCapacity_PowerLawStudentPrior"
# data_model_description = "Source Capacity data model with alpha Prior with alpha=1.4 and a teacher with alpha = 1.2, r = 0.3"
# Sigma_w = power_law_diagonal_matrix(d, 1.4)
# Sigma_delta = np.eye(d)

"""
-------------------
"""


# data_model_type = DataModelType.SourceCapacity
# data_model_name = "SourceCapacity_IDStudentPriorFirstFeatureSigmaDelta"
# data_model_description = "Source Capacity data model with Identity Gaussian Prior and a teacher with alpha = 1.2, r = 0.3, Sigma Delta selecting the first feature only"
# Sigma_w = np.eye(d)
# Sigma_delta = np.diag(np.zeros(d))
# Sigma_delta[0,0] = 1

"""
-------------------
"""


# data_model_type = DataModelType.SourceCapacity
# data_model_name = "SourceCapacity_IDStudentPriorFirstFeatureSigmaDelta"
# data_model_description = "Source Capacity data model with Identity Gaussian Prior and a teacher with alpha = 1.2, r = 0.3, Sigma Delta selecting the first feature only"
# Sigma_w = np.eye(d)
# Sigma_delta = np.diag(np.zeros(d))
# Sigma_delta[0,0] = 1

"""
-------------------
"""


data_model_type = DataModelType.KFeaturesModel
feature_ratios = np.array([0.5,0.5])
features_x = np.array([10,1])
features_theta = np.array([1,1])
data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingIdentity_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}"
data_model_description = "2 Features, Theta Identity, Sigma_delta Identity"
Sigma_w = np.eye(d)
Sigma_delta = np.eye(d)
Sigma_upsilon = 5*np.eye(d)
Sigma_upsilon = Sigma_upsilon * d / np.trace(Sigma_upsilon)
k_features_attacking_identity_protecting_identity_definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)

# theta first strong
data_model_type = DataModelType.KFeaturesModel
feature_ratios = np.array([0.5,0.5])
features_x = np.array([10,1])
features_theta = np.array([10,1])
data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingIdentity_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}"
data_model_description = "2 Features, Theta First Half Strong, Sigma_delta Identity"
Sigma_w = np.eye(d)
Sigma_delta = np.eye(d)
Sigma_upsilon = 5*np.eye(d)
Sigma_upsilon = Sigma_upsilon * d / np.trace(Sigma_upsilon)
k_features_attacking_identity_protecting_identity_theta_first_definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)

# theta second strong
data_model_type = DataModelType.KFeaturesModel
feature_ratios = np.array([0.5,0.5])
features_x = np.array([10,1])
features_theta = np.array([1,10])
data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingIdentity_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}"
data_model_description = "2 Features, Theta Second Half Strong, Sigma_delta Identity"
Sigma_w = np.eye(d)
Sigma_delta = np.eye(d)
Sigma_upsilon = 5*np.eye(d)
Sigma_upsilon = Sigma_upsilon * d / np.trace(Sigma_upsilon)
k_features_attacking_identity_protecting_identity_theta_second_definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)

######### Attacking the first half of the features ##########

data_model_type = DataModelType.KFeaturesModel
feature_ratios = np.array([0.5,0.5])
features_x = np.array([10,1])
features_theta = np.array([1,1])
data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingIdentity_AttackingFirstHalf_{feature_ratios}_{features_x}_{features_theta}"
data_model_description = "2 Features, Theta Identity, Sigma_delta Identity, Sigma_upsilon 10*Identity for the first half of the features"
Sigma_w = np.eye(d)
Sigma_delta = np.eye(d)
Sigma_upsilon = np.zeros((d,d))
Sigma_upsilon[0:int(d/2),0:int(d/2)] = 5*np.eye(int(d/2))
Sigma_upsilon = Sigma_upsilon * d / np.trace(Sigma_upsilon)
k_features_attacking_first_half_protecting_identity_definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)

######### Attacking the second half of the features ##########

data_model_type = DataModelType.KFeaturesModel
feature_ratios = np.array([0.5,0.5])
features_x = np.array([10,1])
features_theta = np.array([1,1])
data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingIdentity_AttackingSecondHalf_{feature_ratios}_{features_x}_{features_theta}"
data_model_description = "2 Features, Theta Identity, Sigma_delta Identity, Sigma_upsilon 10*Identity for the second half of the features"
Sigma_w = np.eye(d)
Sigma_delta = np.eye(d)
Sigma_upsilon = np.zeros((d,d))
Sigma_upsilon[int(d/2):d,int(d/2):d] = 5*np.eye(int(d/2))
Sigma_upsilon = Sigma_upsilon * d / np.trace(Sigma_upsilon)
k_features_attacking_second_half_protecting_identity_definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)


"""
Protecting second half
"""

######### Attacking and protecting the second half of the features ##########

data_model_type = DataModelType.KFeaturesModel
feature_ratios = np.array([0.5,0.5])
features_x = np.array([10,1])
features_theta = np.array([1,1])
data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingSecondHalf_AttackingSecondHalf_{feature_ratios}_{features_x}_{features_theta}_SD_1_10_SU_1_10"
data_model_description = "2 Features, Theta Identity, Sigma_upsilon 10*Identity for the second half of the features, Sigma_delta 10*Identity for the second half of the features"
Sigma_w = np.eye(d)
Sigma_delta = np.eye(d)
Sigma_delta[int(d/2):d,int(d/2):d] = 10*np.eye(int(d/2))
Sigma_upsilon = np.eye(d)
Sigma_upsilon[int(d/2):d,int(d/2):d] = 10*np.eye(int(d/2))
k_features_attacking_second_half_protecting_second_half_definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)

######### Protecting the second half of the features and attacking Identity ##########

data_model_type = DataModelType.KFeaturesModel
feature_ratios = np.array([0.5,0.5])
features_x = np.array([10,1])
features_theta = np.array([1,1])
data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingSecondHalf_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_10_SU_1_10"
data_model_description = "2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta 10*Identity for the second half of the features"
Sigma_w = np.eye(d)
Sigma_delta = np.eye(d)
Sigma_delta[int(d/2):d,int(d/2):d] = 10*np.eye(int(d/2))
Sigma_upsilon = np.eye(d)
k_features_attacking_identity_protecting_second_half_definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)


# ######### Protecting the second half of the features and attacking the first half ##########

data_model_type = DataModelType.KFeaturesModel
feature_ratios = np.array([0.5,0.5])
features_x = np.array([10,1])
features_theta = np.array([1,1])
data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingSecondHalf_AttackingFirstHalf_{feature_ratios}_{features_x}_{features_theta}_SD_1_10_SU_1_10"
data_model_description = "2 Features, Theta Identity, Sigma_upsilon 10*Identity for the first half of the features, Sigma_delta 10*Identity for the first half of the features"
Sigma_w = np.eye(d)
Sigma_delta = np.eye(d)
Sigma_delta[int(d/2):d,int(d/2):d] = 10*np.eye(int(d/2))
Sigma_upsilon = np.eye(d)
Sigma_upsilon[0:int(d/2),0:int(d/2)] = 10*np.eye(int(d/2))
k_features_attacking_first_half_protecting_second_half_definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)

"""
Protecting first half
"""

######### Protecting the first half of the features and attacking the second half ##########

data_model_type = DataModelType.KFeaturesModel
feature_ratios = np.array([0.5,0.5])
features_x = np.array([10,1])
features_theta = np.array([1,1])
data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingFirstHalf_AttackingSecondHalf_{feature_ratios}_{features_x}_{features_theta}_SD_1_10_SU_1_10"
data_model_description = "2 Features, Theta Identity, Sigma_upsilon 10*Identity for the second half of the features, Sigma_delta 10*Identity for the first half of the features"
Sigma_w = np.eye(d)
Sigma_delta = np.eye(d)
Sigma_delta[0:int(d/2),0:int(d/2)] = 10*np.eye(int(d/2))
Sigma_upsilon = np.eye(d)
Sigma_upsilon[int(d/2):d,int(d/2):d] = 10*np.eye(int(d/2))
k_features_attacking_second_half_protecting_first_half_definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)

######### Protecting the first half of the features and attacking Identity ##########

data_model_type = DataModelType.KFeaturesModel
feature_ratios = np.array([0.5,0.5])
features_x = np.array([10,1])
features_theta = np.array([1,1])
data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingFirstHalf_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_10_SU_1_10"
data_model_description = "2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta 10*Identity for the first half of the features"
Sigma_w = np.eye(d)
Sigma_delta = np.eye(d)
Sigma_delta[0:int(d/2),0:int(d/2)] = 10*np.eye(int(d/2))
Sigma_upsilon = np.eye(d)
k_features_attacking_identity_protecting_first_half_definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)


# ######### Protecting the first half of the features and attacking the first half ##########

data_model_type = DataModelType.KFeaturesModel
feature_ratios = np.array([0.5,0.5])
features_x = np.array([10,1])
features_theta = np.array([1,1])
data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingFirstHalf_AttackingFirstHalf_{feature_ratios}_{features_x}_{features_theta}_SD_1_10_SU_1_10"
data_model_description = "2 Features, Theta Identity, Sigma_upsilon 10*Identity for the first half of the features, Sigma_delta 10*Identity for the first half of the features"
Sigma_w = np.eye(d)
Sigma_delta = np.eye(d)
Sigma_delta[0:int(d/2),0:int(d/2)] = 10*np.eye(int(d/2))
Sigma_upsilon = np.eye(d)
Sigma_upsilon[0:int(d/2),0:int(d/2)] = 10*np.eye(int(d/2))
k_features_attacking_first_half_protecting_first_half_definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)


"""
------------------------------------------------------------------------------------------------------------------------
    Organize your data_model_definitions in a list.
------------------------------------------------------------------------------------------------------------------------
"""

data_model_definitions = [vanilla_gaussian_data_model_definition, k_features_attacking_identity_protecting_identity_definition]
data_model_definitions = [vanilla_gaussian_data_model_definition]
data_model_definitions = [k_features_attacking_identity_protecting_first_half_definition]
data_model_definitions = [k_features_attacking_identity_protecting_identity_definition]
data_model_definitions = [k_features_attacking_identity_protecting_second_half_definition]
data_model_definitions = [k_features_attacking_first_half_protecting_first_half_definition]
data_model_definitions = [k_features_attacking_first_half_protecting_identity_definition]
data_model_definitions = [k_features_attacking_first_half_protecting_second_half_definition]
data_model_definitions = [k_features_attacking_second_half_protecting_first_half_definition]
data_model_definitions = [k_features_attacking_second_half_protecting_second_half_definition]
data_model_definitions = [k_features_attacking_second_half_protecting_identity_definition]

memory = "100G"

"""
------------ Combined Experiments
"""

# We want to prouce 4 bunch of colormap plots at high alpha
# First we want to Sweep From Identity to Protecting the First Half with some Psi_10 and plot it against a sweep where we increasingly attack the first half

data_model_definitions = []
data_models_description = "Sweep DataModel Protect First vs Attack First"
for first_protect in np.linspace(1,15,10):
    for first_attack in np.linspace(1,15,10):
        # round both to 4 digits
        first_protect = np.round(first_protect,4)
        first_attack = np.round(first_attack,4)

        data_model_type = DataModelType.KFeaturesModel
        feature_ratios = np.array([0.5,0.5])
        features_x = np.array([10,1])
        features_theta = np.array([1,1])
        data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingFirstHalf_AttackingFirstHalf_{feature_ratios}_{features_x}_{features_theta}_SD_{first_protect}_1_SU_{first_attack}_1"
        data_model_description = f"2 Features, Theta Identity, Sigma_upsilon {first_attack}*Identity for the first half of the features, Sigma_delta {first_protect}*Identity for the first half of the features"
        Sigma_w = np.eye(d)
        Sigma_delta = np.eye(d)
        Sigma_delta[0:int(d/2),0:int(d/2)] = first_protect*np.eye(int(d/2))
        Sigma_upsilon = np.eye(d)
        Sigma_upsilon[0:int(d/2),0:int(d/2)] = first_attack*np.eye(int(d/2))
        definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
        data_model_definitions.append(definition)




### Protect First Attack Identity

data_model_definitions = []
data_models_description = "Sweep DataModel Protect First vs Attack Identity"
for first_protect in np.linspace(1,15,10):
    for identity_attack in np.linspace(1,15,10):
        # round both to 4 digits
        first_protect = np.round(first_protect,4)
        identity_attack = np.round(identity_attack,4)

        data_model_type = DataModelType.KFeaturesModel
        feature_ratios = np.array([0.5,0.5])
        features_x = np.array([10,1])
        features_theta = np.array([1,1])
        data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingFirstHalf_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_{first_protect}_1_SU_{identity_attack}_{identity_attack}"
        data_model_description = f"2 Features, Theta Identity, Sigma_upsilon {identity_attack}*Identity, Sigma_delta {first_protect}*Identity for the first half of the features"
        Sigma_w = np.eye(d)
        Sigma_delta = np.eye(d)
        Sigma_delta[0:int(d/2),0:int(d/2)] = first_protect*np.eye(int(d/2))
        Sigma_upsilon = np.eye(d)* identity_attack
        definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
        data_model_definitions.append(definition)





### Protect Second Attack Identity

data_model_definitions = []
data_models_description = "Sweep DataModel Protect Second vs Attack Identity"
for second_protect in np.linspace(1,15,10):
    for identity_attack in np.linspace(1,15,10):
        # round both to 4 digits
        second_protect = np.round(second_protect,4)
        identity_attack = np.round(identity_attack,4)

        data_model_type = DataModelType.KFeaturesModel
        feature_ratios = np.array([0.5,0.5])
        features_x = np.array([10,1])
        features_theta = np.array([1,1])
        data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingSecondHalf_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_{second_protect}_SU_{identity_attack}_{identity_attack}"
        data_model_description = f"2 Features, Theta Identity, Sigma_upsilon {identity_attack}*Identity, Sigma_delta {second_protect}*Identity for the second half of the features"
        Sigma_w = np.eye(d)
        Sigma_delta = np.eye(d)
        Sigma_delta[int(d/2):d,int(d/2):d] = second_protect*np.eye(int(d/2))
        Sigma_upsilon = np.eye(d)* identity_attack
        definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
        data_model_definitions.append(definition)





### Protect Identity Attack Identity

data_model_definitions = []
data_models_description = "Sweep DataModel Protect Identity vs Attack Identity"
for identity_protect in np.linspace(1,15,10):
    for identity_attack in np.linspace(1,15,10):
        # round both to 4 digits
        identity_protect = np.round(identity_protect,4)
        identity_attack = np.round(identity_attack,4)

        data_model_type = DataModelType.KFeaturesModel
        feature_ratios = np.array([0.5,0.5])
        features_x = np.array([10,1])
        features_theta = np.array([1,1])
        data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingIdentity_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_{identity_protect}_{identity_protect}_SU_{identity_attack}_{identity_attack}"
        data_model_description = f"2 Features, Theta Identity, Sigma_upsilon {identity_attack}*Identity, Sigma_delta {identity_protect}*Identity "
        Sigma_w = np.eye(d)
        Sigma_delta = np.eye(d)* identity_protect
        Sigma_upsilon = np.eye(d)* identity_attack
        definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
        data_model_definitions.append(definition)






### Protect Identity Attack First

data_model_definitions = []
data_models_description = "Sweep DataModel Protect Identity vs Attack First"
for identity_protect in np.linspace(1,15,10):
    for first_attack in np.linspace(1,15,10):
        # round both to 4 digits
        identity_protect = np.round(identity_protect,4)
        first_attack = np.round(first_attack,4)

        data_model_type = DataModelType.KFeaturesModel
        feature_ratios = np.array([0.5,0.5])
        features_x = np.array([10,1])
        features_theta = np.array([1,1])
        data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingIdentity_AttackingFirst_{feature_ratios}_{features_x}_{features_theta}_SD_{identity_protect}_{identity_protect}_SU_{first_attack}_1"
        data_model_description = f"2 Features, Theta Identity, Sigma_upsilon {first_attack}*Identity for the first half, Sigma_delta {identity_protect}*Identity "
        Sigma_w = np.eye(d)
        Sigma_delta = np.eye(d)* identity_protect
        Sigma_upsilon = np.eye(d)
        Sigma_upsilon[0:int(d/2),0:int(d/2)] = first_attack*np.eye(int(d/2))
        definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
        data_model_definitions.append(definition)




### Protect Identity Attack Second

data_model_definitions = []
data_models_description = "Sweep DataModel Protect Identity vs Attack Second"
for identity_protect in np.linspace(1,15,10):
    for second_attack in np.linspace(1,15,10):
        # round both to 4 digits
        identity_protect = np.round(identity_protect,4)
        second_attack = np.round(second_attack,4)

        data_model_type = DataModelType.KFeaturesModel
        feature_ratios = np.array([0.5,0.5])
        features_x = np.array([10,1])
        features_theta = np.array([1,1])
        data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingIdentity_AttackingSecond_{feature_ratios}_{features_x}_{features_theta}_SD_{identity_protect}_{identity_protect}_SU_1_{second_attack}"
        data_model_description = f"2 Features, Theta Identity, Sigma_upsilon {second_attack}*Identity for the second half, Sigma_delta {identity_protect}*Identity "
        Sigma_w = np.eye(d)
        Sigma_delta = np.eye(d)* identity_protect
        Sigma_upsilon = np.eye(d)
        Sigma_upsilon[int(d/2):d,int(d/2):d] = second_attack*np.eye(int(d/2))
        definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
        data_model_definitions.append(definition)



# Then we want to Sweep From Identity to Protecting the Second Half with some Psi_10 and plot it against a sweep where we increasingly attack the second half

data_model_definitions = []
data_models_description = "Sweep DataModel Protect Second vs Attack Second"
for second_protect in np.linspace(1,15,10):
    for second_attack in np.linspace(1,15,10):
        # round both to 4 digits
        second_protect = np.round(second_protect,4)
        second_attack = np.round(second_attack,4)

        data_model_type = DataModelType.KFeaturesModel
        feature_ratios = np.array([0.5,0.5])
        features_x = np.array([10,1])
        features_theta = np.array([1,1])
        data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingSecondHalf_AttackingSecondHalf_{feature_ratios}_{features_x}_{features_theta}_SD_1_{second_protect}_SU_1_{second_attack}"
        data_model_description = f"2 Features, Theta Identity, Sigma_upsilon {second_attack}*Identity for the second half of the features, Sigma_delta {second_protect}*Identity for the second half of the features"
        Sigma_w = np.eye(d)
        Sigma_delta = np.eye(d)
        Sigma_delta[int(d/2):d,int(d/2):d] = second_protect*np.eye(int(d/2))
        Sigma_upsilon = np.eye(d)
        Sigma_upsilon[int(d/2):d,int(d/2):d] = second_attack*np.eye(int(d/2))
        definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
        data_model_definitions.append(definition)



# Then we want to Sweep From Identity to Protecting the First Half with some Psi_10 and plot it against a sweep where we increasingly attack the second half

data_model_definitions = []
data_models_description = "Sweep DataModel Protect First vs Attack Second"
for first_protect in np.linspace(1,15,10):
    for second_attack in np.linspace(1,15,10):
        # round both to 4 digits
        first_protect = np.round(first_protect,4)
        second_attack = np.round(second_attack,4)

        data_model_type = DataModelType.KFeaturesModel
        feature_ratios = np.array([0.5,0.5])
        features_x = np.array([10,1])
        features_theta = np.array([1,1])
        data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingFirstHalf_AttackingSecondHalf_{feature_ratios}_{features_x}_{features_theta}_SD_{first_protect}_1_SU_1_{second_attack}"
        data_model_description = f"2 Features, Theta Identity, Sigma_upsilon {second_attack}*Identity for the second half of the features, Sigma_delta {first_protect}*Identity for the first half of the features"
        Sigma_w = np.eye(d)
        Sigma_delta = np.eye(d)
        Sigma_delta[0:int(d/2),0:int(d/2)] = first_protect*np.eye(int(d/2))
        Sigma_upsilon = np.eye(d)
        Sigma_upsilon[int(d/2):d,int(d/2):d] = second_attack*np.eye(int(d/2))
        definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
        data_model_definitions.append(definition)




# Then we want to Sweep From Identity to Protecting the Second Half with some Psi_10 and plot it against a sweep where we increasingly attack the first half

# data_model_definitions = []
# data_models_description = "Sweep DataModel Protect Second vs Attack First"
# for second_protect in np.linspace(1,15,10):
#     for first_attack in np.linspace(1,15,10):
#         # round both to 4 digits
#         second_protect = np.round(second_protect,4)
#         first_attack = np.round(first_attack,4)
#         data_model_type = DataModelType.KFeaturesModel
#         feature_ratios = np.array([0.5,0.5])
#         features_x = np.array([10,1])
#         features_theta = np.array([1,1])
#         data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingSecondHalf_AttackingFirstHalf_{feature_ratios}_{features_x}_{features_theta}_SD_1_{second_protect}_SU_{first_attack}_1"
#         data_model_description = f"2 Features, Theta Identity, Sigma_upsilon {first_attack}*Identity for the first half of the features, Sigma_delta {second_protect}*Identity for the second half of the features"
#         Sigma_w = np.eye(d)
#         Sigma_delta = np.eye(d)
#         Sigma_delta[int(d/2):d,int(d/2):d] = second_protect*np.eye(int(d/2))
#         Sigma_upsilon = np.eye(d)
#         Sigma_upsilon[0:int(d/2),0:int(d/2)] = first_attack*np.eye(int(d/2))
#         definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
#         data_model_definitions.append(definition)


# data_models_description = "Sweep DataModel Protect Second vs Attack First"


# find_robust_half_definitions = [vanilla_gaussian_data_model_definition,  k_features_attacking_identity_protecting_identity_definition]
# #, k_features_attacking_first_half_protecting_identity_definition, k_features_attacking_second_half_protecting_identity_definition
# # vanilla_gaussian_attack_first_half_data_model_definition, vanilla_gaussian_attack_second_half_data_model_definition,
# data_model_definitions = find_robust_half_definitions
# data_models_description = "FindRobustHalfMedium10"


# vanilla_gaussian_data_model_definition.name = "2_" + vanilla_gaussian_data_model_definition.name
# k_features_attacking_identity_protecting_identity_definition.name = "2_" + k_features_attacking_identity_protecting_identity_definition.name
# compare_unnormalized_Sigma_x_robustness_definitions = [vanilla_gaussian_data_model_definition, vanilla_gaussian_times_10_data_model_definition, k_features_attacking_identity_protecting_identity_definition]
# data_model_definitions = compare_unnormalized_Sigma_x_robustness_definitions
# data_models_description = "CompareUnnormalizedSigmaXRobustness"

# # Testing the core idea by Ilias. Imperceptible (small variance), yet strongly predictive (strong theta zero)
# ilias_core_usefulness_definitions = [vanilla_gaussian_data_model_definition, k_features_attacking_identity_protecting_identity_definition, vanilla_gaussian_theta_first, k_features_attacking_identity_protecting_identity_theta_first_definition, k_features_attacking_identity_protecting_identity_theta_second_definition]
# data_model_definitions = ilias_core_usefulness_definitions
# data_models_description = "IliasCoreUsefulnessEqualThetaSigmaX"

# ilias_core_usefulness_definitions = [vanilla_gaussian_data_model_definition, k_features_attacking_identity_protecting_identity_theta_first_definition, k_features_attacking_identity_protecting_identity_theta_second_definition]
# data_model_definitions = ilias_core_usefulness_definitions
# data_models_description = "Test"


#### Sweep in Theta Ratio

data_model_definitions = []
# data_models_description = "NoNormalisation"
data_models_description = "TraceNormalisationInverseComparison"
# data_models_description = "RhoNormalisationInverseComparison"

for theta_ratio in np.logspace(0,3,20):
    if theta_ratio == 1:
        ra = 1
    else:
        ra = 2

    for i in range(ra):

        if i == 0:
            theta_first = 1
            theta_second = 1/theta_ratio
        else:
            theta_first = 1
            theta_second = theta_ratio

        # round both to 4 digits
        theta_first = np.round(theta_first,4)
        theta_second = np.round(theta_second,4)

        data_model_type = DataModelType.KFeaturesModel
        feature_ratios = np.array([0.5,0.5])
        features_x = np.array([10,1])
        features_theta = np.array([theta_first,theta_second])
        data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingIdentity_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
        data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
        Sigma_w = np.eye(d)
        Sigma_delta = np.eye(d)
        Sigma_upsilon = np.eye(d)
        definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
        data_model_definitions.append(definition)

        data_model_type = DataModelType.KFeaturesModel
        feature_ratios = np.array([0.5,0.5])
        features_x = np.array([0.1,1])
        features_theta = np.array([theta_first,theta_second])
        data_model_name = f"KFeaturesModel_TwoFeatures_ProtectingIdentity_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
        data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
        Sigma_w = np.eye(d)
        Sigma_delta = np.eye(d)
        Sigma_upsilon = np.eye(d)
        definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
        data_model_definitions.append(definition)

        data_model_type = DataModelType.KFeaturesModel
        feature_ratios = np.array([0.5,0.5])
        features_x = np.array([1,1])
        features_theta = np.array([theta_first,theta_second])
        data_model_name = f"VanillaGaussian_ProtectingIdentity_AttackingIdentity_{feature_ratios}_{features_x}_{features_theta}_SD_1_1_SU_1_1"
        data_model_description = f"2 Features, Theta Identity, Sigma_upsilon Identity, Sigma_delta Identity"
        Sigma_w = np.eye(d)
        Sigma_delta = np.eye(d)
        Sigma_upsilon = np.eye(d)
        definition = DataModelDefinition(d,delete_existing,normalize_matrices, Sigma_w, Sigma_delta, Sigma_upsilon, data_model_name, data_model_description, data_model_type, feature_ratios, features_x, features_theta)
        data_model_definitions.append(definition)






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
experiment_filename = f"ThetaSweeps/{data_models_description}/sweep_experiment.json"
# experiment_filename = f"ColorMapSweeps/{data_models_description}/sweep_experiment.json"
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
alphas: np.ndarray = np.array([100000]) # np.logspace(-0.8,1.5,6) #np.linspace(0.1,10,15)  #
epsilons: np.ndarray = np.array([0.0,0.2,0.4,0.6,0.8]) # np.linspace(0,0.6,2) # np.array([0.0,0.2]) # np.array([0,0.1,0.3,0.4,0.5]) 
lambdas: np.ndarray = np.array([1e-3]) #np.logspace(-4,2,15) # np.logspace(-1,2,1) #np.concatenate([-np.logspace(-4,-1,10),np.logspace(-6,-3,2)])  #np.array([-0.0001])
test_against_epsilons: np.ndarray = np.array([0.2,0.4,0.6])
taus: np.ndarray = np.array([0.05, 1])

# round the lambdas, epsilons and alphas for 4 digits
alphas = np.round(alphas,4)    
epsilons = np.round(epsilons,4)
lambdas = np.round(lambdas,4)
taus = np.round(taus,4)
test_against_epsilons = np.round(test_against_epsilons,4)



# calibration
ps: np.ndarray = None # np.array([0.6,0.75,0.9])
dp: float = 0.01

# Type, name and problem
# experiment_type: ExperimentType = ExperimentType.OptimalLambdaAdversarialTestError
experiment_type: ExperimentType = ExperimentType.Sweep
# experiment_type: ExperimentType = ExperimentType.SweepAtOptimalLambdaAdversarialTestError
experiment_name: str = f"{data_models_description}"
problem_types: list[ProblemType] = [ProblemType.Logistic]

# Fair error
gamma_fair_error: float = 0.01



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