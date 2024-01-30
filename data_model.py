import numpy as np
from enum import Enum
import os
import inspect
import sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from data import *
import pickle
from scipy.stats import multivariate_normal
from abc import ABC, abstractmethod

class DataModelType(Enum):
    AbstractModel = 0
    VanillaGaussian = 1
    SourceCapacity = 2
    RandomCovariate = 3
    MarginGaussian = 4
    KFeaturesModel = 5


class SigmaDeltaProcessType(Enum):
    UseContent = 0
    ComputeTeacherOrthogonal = 1
    ComputeTeacherDirection = 2




"""
-------------------- DataSet --------------------
"""
class DataSet():
    def __init__(self, X, y, X_test, y_test, theta) -> None:
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.theta = theta


class AbstractDataModel(ABC):
    """
    Abstract class for data models
    Properties
    ----------
    d: int
        dimension of the space
    gamma: float
        the aspect ratio of the student space to the teacher space p/d = 1
    model_type: DataModelType
        the type of the data model
    Sigma_x: ndarray
        The data covariance
    theta: ndarray
        The teacher weight prior (d,)
    Sigma_w: ndarray
        The student weight prior (d,d)
    Sigma_delta: ndarray
        The student adversarial budget (d,d)
    rho: float
        the teacher-teacher overlap
    PhiPhiT: ndarray
        The Sigma_x^T Theta Theta^T Sigma_x matrix (d,d)
    name: str
        Optional name of the model, if defined. The pickle file name will contain this name. If the file already exists. An exception will be thrown
    description: str
        Optional description of the model, just a text to describe in words what the model is about
    ----------
    Furthermore, the following spectra are computed
    ----------
    spec_PhiPhit: ndarray
        The spectrum of PhiPhiT (d,)
    spec_Sigma_x: ndarray
        The spectrum of Sigma_x (d,)
    spec_Sigma_delta: ndarray
        The spectrum of Sigma_delta (d,)
    ----------
    The following methods need to be overwritten
    ----------
    generate_data(self, n, tau)
        generates n training data X,y, and test data Xtest,y_test and a teacher weight vector w using noise-level tau
    ----------
    The constructor of a general DataModel shall look as follows:
    ----------
    __init__(self, d,p,logger,source_pickle_path="../",delete_existing = False)
        The constructor computes all the necessary spectra, when creating a model, one only needs to define the matrices described above.
        Note when overwriting, you should define the model_type before calling the super constructor
        Once you initialized the model, call _finish_initialization() to compute the spectra and store the pickle
    ----------
    Other Methods
    ----------
    _finish_initialization()
        computes the spectra and stores the pickle
    get_info()
        returns a json with all the information about the model
    ----------
    """
    def __init__(self, d, logger, delete_existing = False, normalize_matrices = True, source_pickle_path="../", name = "", description = "") -> None:
        self.d = d
        self.logger = logger
        self.source_pickle_path = source_pickle_path
        self.normalize_matrices = normalize_matrices
        
        self.logger.info(f"Initializing data model of type {self.model_type.name} with d={d}")
        self.logger.info(f"normalize_matrices: {normalize_matrices}")

        self.gamma = 1

        # set the model_type if it is undefined
        if not hasattr(self, 'model_type'):
            self.model_type = DataModelType.AbstractModel


        self.name = name
        self.description = description

        self.source_pickle = f"{source_pickle_path}data/data_model_{self.model_type.name}_{d}.pkl"
        if name != "":
            self.source_pickle = f"{source_pickle_path}data/data_model_{self.model_type.name}_{d}_name_{name}.pkl"


        # check if a pickle exists
        
        if os.path.isfile(self.source_pickle) and not delete_existing:
            # load self from pickle 
            with open(self.source_pickle, 'rb') as f:
                # assign all the attributes of the pickle to self
                tmp_dict = pickle.load(f)
                for key in [a for a in dir(tmp_dict) if not a.startswith('__') and not a == "get_data" and not a == "logger" and not a == "get_info"]:
                    value = getattr(tmp_dict, key)
                    setattr(self, key, value)
                self.loaded_from_pickle = True
                self.logger.info("loaded self from pickle")
        else:
            self.loaded_from_pickle = False
            self.logger.info("no pickle found")
            if delete_existing:
                self.logger.info("delete_existing is set to True, deleting existing pickle")
                if os.path.isfile(self.source_pickle):
                    os.remove(self.source_pickle)
        
    def _finish_initialization(self):
        """
        computes the spectra and stores the pickle
        """
        self.logger.info("Finishing initialization")
        assumption_1 = self.Sigma_x - self.Sigma_x.T @ np.linalg.inv(self.Sigma_x) @ self.Sigma_x
        min_eigval = np.min(np.linalg.eigvals(assumption_1))
        if min_eigval < 0:
            self.logger.warning(f"Assumption on Schur Complement failed: Matrix was not positive semi-definite; min eigval: {min_eigval}")

        self.logger.info(f"d: {self.d}")


        self.spec_Sigma_x = np.linalg.eigvals(self.Sigma_x)

        # compute and log the ratio of the first eigenvalue to the last eigenvalue
        self.logger.info(f"Pre-Ratio of first to last eigenvalue of Sigma_x: {self.spec_Sigma_x[0] / self.spec_Sigma_x[-1]}")


        # Normalize all the matrices by dividing by the norm of the matrix
        self.logger.info(f"normalize_matrices: {self.normalize_matrices}")
        if self.normalize_matrices:
            self.logger.info("Normalizing the matrices")
            self.Sigma_x = self.Sigma_x / np.trace(self.Sigma_x ) * self.d        
            self.Sigma_w = self.Sigma_w / np.trace(self.Sigma_w) * self.d
            self.Sigma_delta = self.Sigma_delta / np.trace(self.Sigma_delta) * self.d
            self.Sigma_upsilon = self.Sigma_upsilon / np.trace(self.Sigma_upsilon) * self.d



        self.spec_Sigma_x = np.linalg.eigvals(self.Sigma_x)

        # compute and log the ratio of the first eigenvalue to the last eigenvalue
        self.logger.info(f"Ratio of first to last eigenvalue of Sigma_x: {self.spec_Sigma_x[0] / self.spec_Sigma_x[-1]}")
        

    
        # Compute PhiPhiT
        self.PhiPhiT = self.Sigma_x @ self.theta.reshape(self.d,1) @ self.theta.reshape(1,self.d) @ self.Sigma_x.T
        self.rho = self.theta.dot(self.Sigma_x @ self.theta) / self.d


        self.logger.info(f"Norm Sigma_x: {np.trace(self.Sigma_x)}") 
        self.logger.info(f"Norm Sigma_w: {np.trace(self.Sigma_w)}")
        self.logger.info(f"Norm Sigma_delta: {np.trace(self.Sigma_delta)}")
        self.logger.info(f"Norm Sigma_upsilon: {np.trace(self.Sigma_upsilon)}")
        # log entries of Sigma_x
        self.logger.info(f"Sigma_x: {self.Sigma_x}")
        self.logger.info(f"Sigma_w: {self.Sigma_w}")
        self.logger.info(f"Sigma_delta: {self.Sigma_delta}")
        self.logger.info(f"Sigma_upsilon: {self.Sigma_upsilon}")

        # value count the entries of Sigma_x
        self.logger.info(f"Sigma_x value counts: {np.unique(self.Sigma_x, return_counts=True)}")
        self.logger.info(f"Sigma_w value counts: {np.unique(self.Sigma_w, return_counts=True)}")
        self.logger.info(f"Sigma_delta value counts: {np.unique(self.Sigma_delta, return_counts=True)}")
        self.logger.info(f"Sigma_upsilon value counts: {np.unique(self.Sigma_upsilon, return_counts=True)}")


        # log rho
        self.logger.info(f"rho: {self.rho}")

        # Compute FTerm
        self.FTerm = self.Sigma_x @ self.theta.reshape(self.d,1) @ self.theta.reshape(1,self.d) @ self.Sigma_upsilon.T + self.Sigma_upsilon @ self.theta.reshape(self.d,1) @ self.theta.reshape(1,self.d) @ self.Sigma_x.T


        outer_theta = self.theta.reshape(self.d,1) @ self.theta.reshape(1,self.d)
        self.logger.info(f"Norm outer_theta: {np.trace(outer_theta)}")
        # log evs of outer_theta
        self.logger.info(f"outer_theta eigenvalues: {np.linalg.eigvalsh(outer_theta)[0]}")

        # compute the spectra
        self.spec_PhiPhit = np.linalg.eigvalsh(self.PhiPhiT)
        
        self.spec_Sigma_delta = np.linalg.eigvals(self.Sigma_delta)
        self.spec_Sigma_w = np.linalg.eigvals(self.Sigma_w)
        
        self.spec_Sigma_upsilon = np.linalg.eigvals(self.Sigma_upsilon)
        self.spec_FTerm = np.linalg.eigvalsh(self.FTerm)



        # store the pickle
        self.store_self_to_pickle()

    def store_self_to_pickle(self):

        if self.name != "":
            # if the name is defined, and the source pickle exists, throw an exception before storing and thereby overwriting the existing file. This shall prevent accidentally overwriting an existing file
            if os.path.isfile(self.source_pickle):
                raise Exception(f"File {self.source_pickle} already exists. Please delete it or change the name of the model")

        with open(self.source_pickle, "wb") as f:
            pickle.dump(self, f)
            self.logger.info("stored self to pickle")
            self.logger.info(self.get_info())

    def get_info(self):
        info = {
            'data_model': self.name,
            'data_model_type': self.model_type.name,
            'description': self.description,
            'student_dimension': self.d,
            'aspect_ratio': self.gamma,
            'rho': self.rho,
            'Min EV Sigma_w': np.min(self.spec_Sigma_w),
            'Max EV Sigma_w': np.max(self.spec_Sigma_w),
            'Min EV Sigma_delta': np.min(self.spec_Sigma_delta),
            'Max EV Sigma_delta': np.max(self.spec_Sigma_delta),
            'Min EV Sigma_x': np.min(self.spec_Sigma_x),
            'Max EV Sigma_x': np.max(self.spec_Sigma_x),
            'Min EV PhiPhiT': np.min(self.spec_PhiPhit),
            'Max EV PhiPhiT': np.max(self.spec_PhiPhit),
            'Spec Sigma_Upsilon': np.linalg.norm(self.spec_Sigma_upsilon),
            # 'Norm Sigma_w': np.linalg.norm(self.Sigma_w),
            # 'Norm Sigma_delta': np.linalg.norm(self.Sigma_delta),
            # 'Norm Sigma_x': np.linalg.norm(self.Sigma_x),
            # 'Norm PhiPhiT': np.linalg.norm(self.PhiPhiT),
            # ''
        }
        return info
    
    @abstractmethod
    def generate_data(self, n, tau):
        """
        To override
        The data is never normalized, (unless a method is used that forces us to do so)
        The student normalizes the data
        """




"""
-------------------- Vanilla Gaussian Data Model --------------------
"""

class VanillaGaussianDataModel(AbstractDataModel):
    def __init__(self, d, logger, delete_existing = False, normalize_matrices = True,  source_pickle_path="../", Sigma_w = None, Sigma_delta = None, Sigma_upsilon = None, name = "", description = "") -> None:
        self.model_type = DataModelType.VanillaGaussian
        super().__init__(d, logger,delete_existing=delete_existing, normalize_matrices=normalize_matrices, source_pickle_path=source_pickle_path,name=name,description=description)

        if not self.loaded_from_pickle:

            self.Sigma_x = np.eye(self.d)
            self.Sigma_theta = np.eye(self.d)

            self.Sigma_w = Sigma_w
            self.Sigma_delta = Sigma_delta
            self.Sigma_upsilon = Sigma_upsilon
            if self.Sigma_w is None:
                self.Sigma_w = np.eye(self.d)
            if self.Sigma_delta is None:
                self.Sigma_delta = np.eye(self.d)
            if self.Sigma_upsilon is None:
                self.Sigma_upsilon = np.eye(self.d)

            self.rho = 1
            self.PhiPhiT = np.eye(self.d)
            
            self.FTerm = self.Sigma_x.T * self.Sigma_theta * self.Sigma_upsilon + self.Sigma_upsilon.T * self.Sigma_theta * self.Sigma_x

            self._finish_initialization()

    def generate_data(self, n, tau) -> DataSet:
        the = np.random.normal(0,1, self.d)
        # X = np.random.normal(0,1,(n,self.d))
        X = np.random.default_rng().multivariate_normal(np.zeros(self.d), self.Sigma_x, n, method="cholesky")  
        y = np.sign(X @ the / np.sqrt(self.d) + tau * np.random.normal(0,1,(n,))) 
        # X_test = np.random.normal(0,1,(100000,self.d))
        X_test = np.random.default_rng().multivariate_normal(np.zeros(self.d), self.Sigma_x, 100000, method="cholesky")  
        y_test = np.sign(X_test @ the / np.sqrt(self.d)  + tau * np.random.normal(0,1,(100000,)))
        return DataSet(X, y, X_test, y_test, the)
    

class KFeaturesModel(AbstractDataModel):
    def __init__(self, d,logger, delete_existing = False, normalize_matrices = True, attack_equal_defense = False, source_pickle_path="../",Sigma_w_content = None,Sigma_delta_content = None, Sigma_upsilon_content = None, name="", description = "", feature_ratios = None, features_x =None, features_theta = None, process_sigma_type: SigmaDeltaProcessType = SigmaDeltaProcessType.UseContent)->None:
        """
            k = len(feature_ratios)
            feature_ratios = np.array([2,d-2]) # must sum to d and be of length k
            features_x = np.array([100,1]) # must be of length k and contains each features size for the data covariance X
            features_theta = np.array([1,1]) # must be of length k and contains each features size for the teacher prior
        """

        self.d = d
    
        if feature_ratios is None:
            feature_ratios = np.array([1/d,1-1/d])
        if features_x is None:
            features_x = np.array([10,1])
        if features_theta is None:
            features_theta = np.array([1,1])

        self.model_type = DataModelType.KFeaturesModel
        super().__init__(d,logger,delete_existing=delete_existing, normalize_matrices=normalize_matrices, source_pickle_path=source_pickle_path, name=name,description=description)

        if not self.loaded_from_pickle:

            
            k = len(feature_ratios)

            # transform the feature ratios to feature sizes
            feature_sizes = np.floor(feature_ratios * d).astype(int)

            self.theta = np.zeros(d)
            spec_Omega0 = np.zeros(d)
            for i in range(k):
                self.theta[sum(feature_sizes[:i]):sum(feature_sizes[:i+1])] = features_theta[i]
                spec_Omega0[sum(feature_sizes[:i]):sum(feature_sizes[:i+1])] = features_x[i]
            self.Sigma_x=np.diag(spec_Omega0)

            self.feature_sizes = feature_sizes

            self.Sigma_theta = np.diag(self.theta)
            self.theta = np.random.default_rng().multivariate_normal(np.zeros(self.d), self.Sigma_theta, method="cholesky") 
            # log theta
            self.logger.info(f"theta: {self.theta}")

            self.rho = self.theta.dot(self.Sigma_x @ self.theta) / d
            # reshape theta
            self.PhiPhiT = self.Sigma_x @ self.theta.reshape(d,1) @ self.theta.reshape(1,d) @ self.Sigma_x.T

            sigma_w = np.zeros(d)
            sigma_delta = np.zeros(d)
            sigma_upsilon = np.zeros(d)

            self.logger.info(f"d: {d}")
            self.logger.info(f"feature_sizes: {feature_sizes}")
            self.logger.info(f"feature_ratios: {feature_ratios}")
            self.logger.info(f"features_x: {features_x}")
            self.logger.info(f"features_theta: {features_theta}")
            self.logger.info(f"Sigma_w_content: {Sigma_w_content}")
            self.logger.info(f"Sigma_delta_content: {Sigma_delta_content}")
            self.logger.info(f"Sigma_upsilon_content: {Sigma_upsilon_content}")
            

            for i in range(k):
                sigma_w[sum(feature_sizes[:i]):sum(feature_sizes[:i+1])] = Sigma_w_content[i]
                sigma_delta[sum(feature_sizes[:i]):sum(feature_sizes[:i+1])] = Sigma_delta_content[i]
                sigma_upsilon[sum(feature_sizes[:i]):sum(feature_sizes[:i+1])] = Sigma_upsilon_content[i]
            self.Sigma_w = np.diag(sigma_w)
            self.Sigma_delta = np.diag(sigma_delta)
            self.Sigma_upsilon = np.diag(sigma_upsilon)
            
            self.V_i = np.ones(d)

            if process_sigma_type == SigmaDeltaProcessType.ComputeTeacherOrthogonal or process_sigma_type == SigmaDeltaProcessType.ComputeTeacherDirection:

                if process_sigma_type == SigmaDeltaProcessType.ComputeTeacherOrthogonal:
                    vprime = np.random.normal(0,1,d)

                    # chose v = vprime - <vprime,theta> theta / ||theta||^2

                    v = vprime - np.dot(vprime,self.theta) * self.theta / np.linalg.norm(self.theta)**2

                    # normalize v
                    v = v / np.linalg.norm(v)

                    Sigma_delta_content = v

                    # log in this case v dot theta
                    self.logger.info(f"v dot theta: {np.dot(v,self.theta)}")

                elif process_sigma_type == SigmaDeltaProcessType.ComputeTeacherDirection:
                    v = self.theta
                    v = v/np.linalg.norm(v)                    
                    Sigma_delta_content = v

                    # log in this case v dot theta
                    self.logger.info(f"v dot theta: {np.dot(v,self.theta)}")
                    

                # Only for the Optimal Defense Experiment
                self.Sigma_delta = np.outer(Sigma_delta_content, Sigma_delta_content)
                
                # sample a random covariance matrix
                random_matrix = np.random.normal(0,0.0001,(d,d))
                self.Sigma_delta = random_matrix.T @ random_matrix + self.Sigma_delta * 10000
                
                self.V_i = Sigma_delta_content

            if attack_equal_defense:
                self.Sigma_upsilon = self.Sigma_delta

            # log the eigenvalues of Sigma_delta
            self.logger.info(f"Sigma_delta eigenvalues: {np.linalg.eigvals(self.Sigma_delta)}")


            if self.Sigma_w is None:
                self.Sigma_w = np.eye(self.d)
            if self.Sigma_delta is None:
                self.Sigma_delta = np.eye(self.d)
            if self.Sigma_upsilon is None:
                self.Sigma_upsilon = np.eye(self.d)

            self.FTerm = self.Sigma_x @ self.theta.reshape(d,1) @ self.theta.reshape(1,d) @ self.Sigma_upsilon.T + self.Sigma_upsilon @ self.theta.reshape(d,1) @ self.theta.reshape(1,d) @ self.Sigma_x.T

            self._finish_initialization()


    def generate_data(self, n, tau) -> DataSet:

        theta = self.theta
        X = np.random.default_rng().multivariate_normal(np.zeros(self.d), self.Sigma_x, n, method="cholesky")  
        
        y = np.sign(X @ theta / np.sqrt(self.d) + tau * np.random.normal(0,1,(n,)))
        
        X_test = np.random.default_rng().multivariate_normal(np.zeros(self.d), self.Sigma_x, 10000, method="cholesky") 
        y_test = np.sign(X_test @ theta / np.sqrt(self.d) + tau * np.random.normal(0,1,(10000,)))

        return DataSet(X, y, X_test, y_test, theta)
    



    

class SourceCapacityDataModel(AbstractDataModel):
    def __init__(self, d,logger, delete_existing = False, normalize_matrices = True, source_pickle_path="../",Sigma_w = None,Sigma_delta = None, Sigma_upsilon = None, name="", description = "")->None:

        self.model_type = DataModelType.SourceCapacity
        super().__init__(d,logger,delete_existing=delete_existing, normalize_matrices=normalize_matrices, source_pickle_path=source_pickle_path, name=name,description=description)

        if not self.loaded_from_pickle:

            alph = 1.2
            r = 0.3

            spec_Omega0 = np.array([self.d/(k+1)**alph for k in range(self.d)])
            self.Sigma_x=np.diag(spec_Omega0)

            theta = np.sqrt(np.array([1/(k+1)**((1+alph*(2*r-1))) for k in range(self.d)]))

            self.rho = np.mean(spec_Omega0 * theta**2) 
            self.PhiPhiT = np.diag(spec_Omega0**2 * theta**2)
            self.Sigma_theta = np.diag(theta**2)


            self.Sigma_w = Sigma_w
            self.Sigma_delta = Sigma_delta
            self.Sigma_upsilon = Sigma_upsilon
            if self.Sigma_w is None:
                self.Sigma_w = np.eye(self.d)
            if self.Sigma_delta is None:
                self.Sigma_delta = np.eye(self.d)
            if self.Sigma_upsilon is None:
                self.Sigma_upsilon = np.eye(self.d)

            self.FTerm = self.Sigma_x.T * self.Sigma_theta * self.Sigma_upsilon + self.Sigma_upsilon.T * self.Sigma_theta * self.Sigma_x

            self._finish_initialization()


    def generate_data(self, n, tau) -> DataSet:
        theta = np.random.default_rng().multivariate_normal(np.zeros(self.d), self.Sigma_theta, 1, method="cholesky")[0]  
        X = np.random.default_rng().multivariate_normal(np.zeros(self.d), self.Sigma_x, n, method="cholesky")  
        
        y = np.sign(X @ theta / np.sqrt(self.d) + tau * np.random.normal(0,1,(n,)))
        
        X_test = np.random.default_rng().multivariate_normal(np.zeros(self.d), self.Sigma_x, 10000, method="cholesky") 
        y_test = np.sign(X_test @ theta / np.sqrt(self.d) + tau * np.random.normal(0,1,(10000,)))

        return DataSet(X, y, X_test, y_test, theta)
    

class MarginGaussianDataModel(AbstractDataModel):
    def __init__(self, d, logger, delete_existing = False, normalize_matrices = True, source_pickle_path="../", Sigma_w = None, Sigma_delta = None, Sigma_upsilon = None, name = "", description = "") -> None:
        self.model_type = DataModelType.MarginGaussian
        super().__init__(d, logger,delete_existing=delete_existing, normalize_matrices=normalize_matrices, source_pickle_path=source_pickle_path,name=name,description=description)

        if not self.loaded_from_pickle:

            """
            Warning, these matrices can be anything as this model is not meant to work with the state evolution as it is a mixed gaussian model!
            """

            self.Sigma_x = np.eye(self.d)
            self.Sigma_theta = np.eye(self.d)

            self.Sigma_w = Sigma_w
            self.Sigma_delta = Sigma_delta
            self.Sigma_upsilon = Sigma_upsilon
            if self.Sigma_w is None:
                self.Sigma_w = np.eye(self.d)
            if self.Sigma_delta is None:
                self.Sigma_delta = np.eye(self.d)
            if self.Sigma_upsilon is None:
                self.Sigma_upsilon = np.eye(self.d)

            self.FTerm = self.Sigma_x.T * self.Sigma_theta * self.Sigma_upsilon + self.Sigma_upsilon.T * self.Sigma_theta * self.Sigma_x

            self.rho = 1
            self.PhiPhiT = np.eye(self.d)

            self._finish_initialization()

    def generate_data(self, n, tau) -> DataSet:
        # We fix the teacher to be the first eigenvector in a d-dimensional space
        the = np.zeros(self.d)
        the[0] = 1
        r = 2.0
        
        
        # create the labels
        n_half = np.floor(n/2)
        # convert to int
        n_half = n_half.astype(int)
        n_test_half = np.floor(100000/2)
        n_test_half = n_test_half.astype(int)
        y = np.concatenate([np.ones(n_half),-np.ones(n_half)])
        y_test = np.concatenate([np.ones(n_test_half),-np.ones(n_test_half)])


        # create the data
        X = np.random.normal(0,1,(n_half*2,self.d))
        X_test = np.random.normal(0,1,(n_test_half*2,self.d))

        # change the first dimension of each dataset according to r*y*the
        X[:,0] = r*y*the[0] 
        X_test[:,0] = r*y_test*the[0]
        
        return DataSet(X, y, X_test, y_test, the)


class RandomCovariateDataModel(AbstractDataModel):
    def __init__(self, d,logger, source_pickle_path="../",delete_existing=False)->None:
        self.model_type = DataModelType.RandomCovariate
        super().__init__(d,logger, source_pickle_path,delete_existing)

        raise NotImplementedError("RandomCovariateDataModel is only partially implemented, do not use it yet.")


        self.Sigma_x = np.random.normal(0,0.5,(self.d,self.d))
        self.Sigma_x = self.Sigma_x.T @ self.Sigma_x + np.eye(self.d)
        self.Sigma_theta = np.random.normal(0,0.8,(self.d,self.d))
        self.Sigma_theta = self.Sigma_theta.T @ self.Sigma_theta + np.eye(self.d)
        self.Sigma_w = np.eye(self.d)
        self.Sigma_delta = np.eye(self.d)

        # Let's test if Sigma_x is positive definite
        min_eigval = np.min(np.linalg.eigvalsh(self.Sigma_x))
        if min_eigval < 0:
            raise Exception("Sigma_x is not positive definite; min eigval: ", min_eigval)

        spec_Sigma_x = np.linalg.eigvalsh(self.Sigma_x)
        spec_Sigma_theta = np.linalg.eigvalsh(self.Sigma_theta)

        self.rho = spec_Sigma_x.dot(spec_Sigma_theta) / self.d
        self.PhiPhiT = self.Sigma_x @ self.Sigma_theta @ self.Sigma_x

        self._finish_initialization()

    def generate_data(self, n, tau) -> DataSet:
        the = np.random.default_rng().multivariate_normal(np.zeros(self.d), self.Sigma_theta, method="cholesky")
        
        X = np.random.default_rng().multivariate_normal(np.zeros(self.d), self.Sigma_x, n, method="cholesky")            
        y = np.sign(X @ the / np.sqrt(self.d)  + tau * np.random.normal(0,1,(n,)))
        
        X_test = np.random.default_rng().multivariate_normal(np.zeros(self.d), self.Sigma_x, 10000, method="cholesky") 
        
        y_test = np.sign(X_test @ the / np.sqrt(self.d) + tau * np.random.normal(0,1,(10000,)))
        return DataSet(X, y, X_test, y_test, the)
