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
    Sigma_theta: ndarray
        The teacher weight prior (p,p)
    Sigma_w: ndarray
        The student weight prior (d,d)
    Sigma_delta: ndarray
        The student adversarial budget (d,d)
    rho: float
        the teacher-teacher overlap
    PhiPhiT: ndarray
        The Sigma_x^T Theta Theta^T Sigma_x matrix (d,d)
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
    def __init__(self, d, logger, delete_existing = False, source_pickle_path="../") -> None:
        self.d = d
        self.logger = logger
        self.source_pickle_path = source_pickle_path

        self.gamma = 1

        # set the model_type if it is undefined
        if not hasattr(self, 'model_type'):
            self.model_type = DataModelType.AbstractModel


        # check if a pickle exists
        self.source_pickle = f"{source_pickle_path}data/data_model_{self.model_type.name}_{d}.pkl"
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
        
    def _finish_initialization(self):
        """
        computes the spectra and stores the pickle
        """
        self.logger.info("Finishing initialization")
        assumption_1 = self.Sigma_x - self.Sigma_x.T @ np.linalg.inv(self.Sigma_x) @ self.Sigma_x
        min_eigval = np.min(np.linalg.eigvalsh(assumption_1))
        if min_eigval < 0:
            self.logger.warning(f"Assumption on Schur Complement failed: Matrix was not positive semi-definite; min eigval: {min_eigval}")

        # compute the spectra
        self.spec_PhiPhit = np.linalg.eigvalsh(self.PhiPhiT)
        self.spec_Sigma_x = np.linalg.eigvalsh(self.Sigma_x)
        self.spec_Sigma_delta = np.linalg.eigvalsh(self.Sigma_delta)
        self.spec_Sigma_w = np.linalg.eigvalsh(self.Sigma_w)

        # store the pickle
        self.store_self_to_pickle()

    def store_self_to_pickle(self):
        with open(self.source_pickle, "wb") as f:
            pickle.dump(self, f)
            self.logger.info("stored self to pickle")

    def get_info(self):
        info = {
            'data_model': 'custom',
            'student_dimension': self.d,
            'aspect_ratio': self.gamma,
            'rho': self.rho,
            'commute': self.commute,
            'kitchen_kind': self.kitchen_kind.name,
            'Norm Sigma_w': np.linalg.norm(self.Sigma_w),
            'Norm Sigma_delta': np.linalg.norm(self.Sigma_delta),
            'Norm Sigma_x': np.linalg.norm(self.Sigma_x),
            'Norm Sigma_theta': np.linalg.norm(self.Sigma_theta),
            'Norm PhiPhiT': np.linalg.norm(self.PhiPhiT),
            'Norm theta': np.linalg.norm(self.theta),
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
-------------------- DataSet --------------------
"""
class DataSet():
    def __init__(self, X, y, X_test, y_test, theta) -> None:
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.theta = theta


"""
-------------------- Vanilla Gaussian Data Model --------------------
"""

class VanillaGaussianDataModel(AbstractDataModel):
    def __init__(self, d, logger, delete_existing = False, source_pickle_path="../", Sigma_w = None, Sigma_delta = None) -> None:
        self.model_type = DataModelType.VanillaGaussian
        super().__init__(d, logger,delete_existing=delete_existing, source_pickle_path=source_pickle_path)

        if not self.loaded_from_pickle:

            self.Sigma_x = np.eye(self.d)
            self.Sigma_theta = np.eye(self.d)

            self.Sigma_w = Sigma_w
            self.Sigma_delta = Sigma_delta
            if self.Sigma_w is None:
                self.Sigma_w = np.eye(self.d)
            if self.Sigma_delta is None:
                self.Sigma_delta = np.eye(self.d)

            self.rho = 1
            self.PhiPhiT = np.eye(self.d)

            self._finish_initialization()

    def generate_data(self, n, tau) -> DataSet:
        the = np.random.normal(0,1, self.d)
        c = np.random.normal(0,1,(n,self.d))
        u = c
        y = np.sign(u @ the / np.sqrt(self.d) + tau * np.random.normal(0,1,(n,))) 
        X = u
        X_test = np.random.normal(0,1,(100000,self.d))
        y_test = np.sign(X_test @ the  / np.sqrt(self.d) + tau * np.random.normal(0,1,(100000,)))
        return DataSet(X, y, X_test, y_test, the)
    
class SourceCapacityDataModel(AbstractDataModel):
    def __init__(self, d,logger, delete_existing = False, source_pickle_path="../",Sigma_w = None,Sigma_delta=None)->None:

        self.model_type = DataModelType.SourceCapacity
        super().__init__(d,logger,delete_existing=delete_existing, source_pickle_path=source_pickle_path)

        if not self.loaded_from_pickle:

            alph = 1.2
            r = 0.3

            spec_Omega0 = np.array([self.d/(k+1)**alph for k in range(self.d)])
            self.Sigma_x=np.diag(spec_Omega0)

            self.theta = np.sqrt(np.array([1/(k+1)**((1+alph*(2*r-1))) for k in range(self.d)]))

            self.rho = np.mean(spec_Omega0 * self.theta**2) 
            self.PhiPhiT = np.diag(spec_Omega0**2 * self.theta**2)
            self.Sigma_theta = np.diag(self.theta**2)


            self.Sigma_w = Sigma_w
            self.Sigma_delta = Sigma_delta
            if self.Sigma_w is None:
                self.Sigma_w = np.eye(self.d)
            if self.Sigma_delta is None:
                self.Sigma_delta = np.eye(self.d)

            self._finish_initialization()


    def generate_data(self, n, tau) -> DataSet:

        X = np.random.default_rng().multivariate_normal(np.zeros(self.d), self.Sigma_x, n, method="cholesky")  
        
        y = np.sign(X @ self.theta / np.sqrt(self.d) + tau * np.random.normal(0,1,(n,)))
        
        X_test = np.random.default_rng().multivariate_normal(np.zeros(self.d), self.Sigma_x, 10000, method="cholesky") 
        y_test = np.sign(X_test @ self.theta / np.sqrt(self.d) + tau * np.random.normal(0,1,(10000,)))

        return DataSet(X, y, X_test, y_test, self.theta)
    

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
