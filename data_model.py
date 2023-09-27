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


# Taken from Loureiro

class DataModelType(Enum):
    Undefined = 0
    Gaussian = 1
    FashionMNIST = 2


class DataModel(object):
    '''
    Base data model.
    '''
    def __init__(self):
        self.p, self.d = self.Phi.shape

        self._diagonalise()
        self._commute()
        self.DataModelType = DataModelType.Undefined

    def get_info(self):
        info = {
            'data_model': 'base',
            'latent_dimension': self.d,
            'input_dimension': self.p
        }
        return info

    def _check_commute(self):
        if np.linalg.norm(self.Omega @ self.PhiPhiT - self.PhiPhiT @ self.Omega) < 1e-10:
            self.commute = True
        else:
            self.commute = False
            self._UTPhiPhiTU = np.diagonal(self.eigv_Omega.T @ self.PhiPhiT @ self.eigv_Omega)

    def _diagonalise(self):
        '''
        Diagonalise covariance matrices.
        '''
        self.spec_Omega, self.eigv_Omega = np.linalg.eigh(self.Omega)
        self.spec_Omega = np.real(self.spec_Omega)

        self.spec_PhiPhit = np.real(np.linalg.eigvalsh(self.PhiPhiT))

    
    def get_data():
        return None,None,None,None


class Custom(DataModel):
    '''
    Custom allows for user to pass his/her own covariance matrices.
    -- args --
    teacher_teacher_cov: teacher-teacher covariance matrix (Psi)
    student_student_cov: student-student covariance matrix (Omega)
    teacher_student_cov: teacher-student covariance matrix (Phi)
    teacher_weights: teacher weight vector (theta0)
    '''
    def __init__(self, *, teacher_teacher_cov, student_student_cov, 
                 teacher_student_cov, teacher_weights):
        
        self.DataModelType = DataModelType.Undefined

        self.Psi = teacher_teacher_cov
        self.Omega = student_student_cov
        self.Phi = teacher_student_cov.T
        self.theta = teacher_weights
        
        self.p, self.k = self.Phi.shape
        self.gamma = self.k / self.p
        
        self.PhiPhiT = (self.Phi @ self.theta.reshape(self.k,1) @ 
                        self.theta.reshape(1,self.k) @ self.Phi.T)
        
        self.rho = self.theta.dot(self.Psi @ self.theta) / self.k

        self._check_sym()
        self._diagonalise() # see base_data_model
        self._check_commute()

    def get_info(self):
        info = {
            'data_model': 'custom',
            'teacher_dimension': self.k,
            'student_dimension': self.p,
            'aspect_ratio': self.gamma,
            'rho': self.rho
        }
        return info

    def _check_sym(self):
        '''
        Check if input-input covariance is a symmetric matrix.
        '''
        if (np.linalg.norm(self.Omega - self.Omega.T) > 1e-5):
            print('Student-Student covariance is not a symmetric matrix. Symmetrizing!')
            self.Omega = .5 * (self.Omega+self.Omega.T)

        if (np.linalg.norm(self.Psi - self.Psi.T) > 1e-5):
            print('Teacher-teaccher covariance is not a symmetric matrix. Symmetrizing!')
            self.Psi = .5 * (self.Psi+self.Psi.T)


class CustomSpectra(DataModel):
    '''
    Custom allows for user to pass directly the spectra of the covarinces.
    -- args --
    spec_Psi: teacher-teacher covariance matrix (Psi)
    spec_Omega: student-student covariance matrix (Omega)
    diagonal_term: projection of student-teacher covariance into basis of Omega
    '''
    def __init__(self, *, rho, spec_Omega, diagonal_term, gamma):
        self.rho = rho
        self.spec_Omega = spec_Omega
        self._UTPhiPhiTU = diagonal_term

        self.p = len(self.spec_Omega)
        self.gamma = gamma
        self.k = int(self.gamma * self.p)

        self.commute = False

    def get_info(self):
        info = {
            'data_model': 'custom_spectra',
            'teacher_dimension': self.k,
            'student_dimension': self.p,
            'aspect_ratio': self.gamma,
            'rho': self.rho
        }
        return info
    

class GaussianDataModel(DataModel):

    def __init__(self, d):
        self.DataModelType = DataModelType.Gaussian

        # Psi 
        self.Psi = np.eye(d)
        self.Omega = np.eye(d)
        # Phi is a zero matrix
        self.Phi = np.zeros((d,d))
        self.theta = sample_weights(d)
        
        self.p, self.k = self.Phi.shape
        self.gamma = self.k / self.p
        
        self.PhiPhiT = (self.Phi @ self.theta.reshape(self.k,1) @ 
                        self.theta.reshape(1,self.k) @ self.Phi.T)
        
        self.rho = self.theta.dot(self.Psi @ self.theta) / self.k

        self._diagonalise() # see base_data_model (should not be necessary)
        self._check_commute()


    def get_data(self, n, tau):
        Xtrain, y = sample_training_data(self.theta,self.d,n,tau)
        n_test = 100000
        Xtest,ytest = sample_training_data(self.theta,self.d,n_test,tau)
        return Xtrain, y, Xtest, ytest
    
    def get_info(self):
        info = {
            'data_model': 'gaussian',
            'teacher_dimension': self.d,
            'student_dimension': self.d,
            'aspect_ratio': 1,
            'rho': 1
        }
        return info

        
        
class FashionMNISTDataModel(DataModel):

    def __init__(self):

        ntot = 12000

        self.p = ntot
        self.d = 784
        
        self.gamma = self.p/self.d


        # let's see if the pickle file exists
        if os.path.isfile('../data/fashion_mnist.pkl'):
            # Let's read the pickle file
            with open('../data/fashion_mnist.pkl', 'rb') as f:
                data = pickle.load(f)
                X_train = np.array(data['X_train'])
                y_train = np.array(data['y_train'])
                X_test = np.array(data['X_test'])
                y_test = np.array(data['y_test'])
                Omega = np.array(data['Omega'])
                rho = data['rho']
                spec_Omega = np.array(data['spec_Omega'])
                diagUtPhiPhitU = np.array(data['diagUtPhiPhitU'])

        else:

            X_train, y_train = self.load_mnist('../data/fashion', kind='train')
            X_test, y_test = self.load_mnist('../data/fashion', kind='t10k')

            # Let's use a subset of the data and learn to distinguish between two classes
            # Let's pick t-shirt 0 from sneaker 7
            X_train = X_train[np.logical_or(y_train == 0, y_train == 7)]
            y_train = y_train[np.logical_or(y_train == 0, y_train == 7)]
            X_test = X_test[np.logical_or(y_test == 0, y_test == 7)]
            y_test = y_test[np.logical_or(y_test == 0, y_test == 7)]

            # Subtract the mean and divide by the standard deviation
            X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
            X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)

            # change the datatype to float64
            X_train = X_train.astype(np.float64)
            X_test = X_test.astype(np.float64)
            y_test = y_test.astype(np.float64)
            y_train = y_train.astype(np.float64)

            # Make the labels binary
            y_train[y_train == 0] = -1
            y_train[y_train == 7] = 1
            y_test[y_test == 0] = -1
            y_test[y_test == 7] = 1

        

            Omega = X_train.T @ X_train / ntot # student-student
            rho = y_train.dot(y_train) / ntot
            spec_Omega, U = np.linalg.eigh(Omega)
            diagUtPhiPhitU = np.diag(1/ntot * U.T @ X_train.T @ y_train.reshape(ntot,1) @ 
                            y_train.reshape(1,ntot) @ X_train @ U)

        self.rho = rho
        self.spec_Omega = spec_Omega
        self._UTPhiPhiTU = diagUtPhiPhitU

        self.p = len(self.spec_Omega)
        self.k = int(self.gamma * self.p)

        self.commute = False

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_data(self, n, tau):
        if n > self.X_train.shape[0]:
            raise ValueError("n should be smaller than the number of training points")
        Xtrain = self.X_train[:n]
        y = self.y_train[:n]
        return Xtrain, y, self.X_test, self.y_test

    def load_mnist(self, path, kind='train'):
        import os
        import gzip
        import numpy as np

        """Load MNIST data from `path`"""
        labels_path = os.path.join(path,
                                '%s-labels-idx1-ubyte.gz'
                                % kind)
        images_path = os.path.join(path,
                                '%s-images-idx3-ubyte.gz'
                                % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                offset=16).reshape(len(labels), 784)

        return images, labels

    def get_info(self):
        info = {
            'data_model': 'fashion_mnist',
            'teacher_dimension': self.k,
            'student_dimension': self.p,
            'aspect_ratio': self.gamma,
            'rho': self.rho
        }
        return info
    
