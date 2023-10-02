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



# Taken from Loureiro

class DataModelType(Enum):
    Undefined = 0
    Gaussian = 1
    FashionMNIST = 2
    RandomKitchenSink = 3


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

        # assume iid gaussian prior
        self.Sigma_w = np.eye(self.k)
        # Assign the inverse of Sigma_w
        self.Sigma_w_inv = np.linalg.inv(self.Sigma_w)
        
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

        self.Sigma_w = np.eye(d)
        self.Sigma_w_inv = np.linalg.inv(self.Sigma_w)
        
        self.p, self.k = self.Phi.shape
        self.gamma = self.k / self.p
        
        self.PhiPhiT = (self.Phi @ self.theta.reshape(self.k,1) @ 
                        self.theta.reshape(1,self.k) @ self.Phi.T)
        
        self.rho = self.theta.dot(self.Psi @ self.theta) / self.k

        self._diagonalise() # see base_data_model (should not be necessary)
        self._check_commute()


    def get_data(self, n, tau):
        raise NotImplementedError("Gaussian data model does not have data")
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

        source_pickle = "../data/fashion_mnist.pkl"
        neural_source_pickle = "../data/neural_fashion_mnist.pkl"

        source_file = neural_source_pickle

        # let's see if the pickle file exists
        if os.path.isfile(source_file):
            # Let's read the pickle file
            with open(source_file, 'rb') as f:
                data = pickle.load(f)
                X_train = np.array(data['X_train'])
                y_train = np.array(data['y_train'])
                X_test = np.array(data['X_test'])
                y_test = np.array(data['y_test'])
                Omega = np.array(data['Omega'])
                rho = data['rho']
                spec_Omega = np.array(data['spec_Omega'])
                diagUtPhiPhitU = np.array(data['diagUtPhiPhitU'])

                ntot = X_train.shape[0]
                self.d = X_train.shape[1]

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

        self.Sigma_w = np.eye(d) # Just an assumption...
        self.Sigma_w_inv = np.linalg.inv(self.Sigma_w)
        

        self.p = ntot
        
        
        self.gamma = self.p/self.d
    
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
        raise NotImplementedError("Fashion MNIST data model does not have data")
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
    

class KitchenKind(Enum):
    Vanilla = 1
    StudentOnly = 2
    TeacherStudent = 3
    DoubleCovariate = 4




def sample_multivariate_gaussian(mean, covariance_matrix, n):
    """
    Sample from a multivariate Gaussian distribution with zero mean and a given covariance matrix.
    
    Args:
    - mean: The mean vector (list or numpy array).
    - covariance_matrix: The covariance matrix (list or numpy array).
    - n: The number of samples to generate.
    
    Returns:
    - samples: An array of shape (n, k), where k is the dimensionality of the distribution.
    """
    if len(mean) != len(covariance_matrix) or len(covariance_matrix[0]) != len(mean):
        raise ValueError("The dimensions of 'mean' and 'covariance_matrix' must match.")

    k = len(mean)  # Dimensionality of the distribution

    # Cholesky decomposition of the covariance matrix
    L = np.linalg.cholesky(covariance_matrix)

    # Generate independent standard normal samples
    z = np.random.randn(n, k)

    # Transform the standard normal samples to the desired distribution
    samples = mean + np.dot(z, L.T)

    return samples



class RandomKitchenSinkDataModel(DataModel):
    def __init__(self, student_dimension, teacher_dimension, logger, source_pickle_path = "../", delete_existing = False):  

        self.logger = logger
 
        logger.info("Let that Random Kitchen Sink in")

        self.kitchen_kind = KitchenKind.DoubleCovariate

        self.d = student_dimension
        self.p = teacher_dimension



        # check if a pickle exists
        source_pickle = f"{source_pickle_path}data/random_kitchen_sink_{student_dimension}_{teacher_dimension}_{self.kitchen_kind.name}.pkl"
        if os.path.isfile(source_pickle) and not delete_existing:
        
            # load self from pickle 
            with open(source_pickle, 'rb') as f:
                # assign all the attributes of the pickle to self
                tmp_dict = pickle.load(f)
                for key in [a for a in dir(tmp_dict) if not a.startswith('__') and not a == "get_data" and not a == "logger" and not a == "get_info"]:
                    value = getattr(tmp_dict, key)
                    # print("setting " + key + " to " + str(value) + " from pickle")
                    setattr(self, key, value)


        else:

            COEFICIENTS = {'relu': (1/np.sqrt(2*np.pi), 0.5, np.sqrt((np.pi-2)/(4*np.pi))), 
                'erf': (0, 2/np.sqrt(3*np.pi), 0.200364), 'tanh': (0, 0.605706, 0.165576),
                'sign': (0, np.sqrt(2/np.pi), np.sqrt(1-2/np.pi))}
            
            if self.kitchen_kind == KitchenKind.Vanilla:
            # ----------------- The Vanilla Gaussian Model ----------------
                assert self.p == self.d, "p must be equal to d for the vanilla gaussian model"
                self.Psi = np.eye(self.p)
                self.Omega = np.eye(self.d)
                self.Phi = np.eye(self.d)
                self.theta = np.random.normal(0,1, self.p) 
                self.Sigma_w = np.eye(self.p)
                self.Sigma_w_inv = np.linalg.inv(self.Sigma_w)

            elif self.kitchen_kind == KitchenKind.TeacherStudent:
            # ----------------- The Teacher and Student Kitchen Sink ----------------

                D = student_dimension + teacher_dimension # dimension of c
                d = student_dimension # dimension of x
                p = teacher_dimension # dimension of k

                self.F_teacher = np.random.normal(0,1, (p,D)) / np.sqrt(D) # teacher random projection
                self.F_student = np.random.normal(0,1, (d,D)) / np.sqrt(D) # student random projection

                # Coefficients
                _, kappa1_teacher, kappastar_teacher = COEFICIENTS['sign']
                _, kappa1_student, kappastar_student = COEFICIENTS['sign']

                # Covariances
                self.Psi = (kappa1_teacher**2 * self.F_teacher @ self.F_teacher.T + kappastar_teacher**2 * np.identity(p))
                self.Omega = (kappa1_student**2 * self.F_student @ self.F_student.T + kappastar_student**2 * np.identity(d))
                self.Phi = kappa1_teacher * kappa1_student * self.F_teacher @ self.F_student.T 

                self.Phi = self.Phi.T # Loureiro transp

                # Print all shapes
                # print('F_teacher', self.F_teacher.shape)
                # print('F_student', self.F_student.shape)
                # print('Psi', Psi.shape)
                # print('Omega', Omega.shape)
                # print('Phi', Phi.shape)

                # Teacher weights
                self.theta = np.random.normal(0,1, self.p) 
                self.Sigma_w = np.eye(self.p)
                self.Sigma_w_inv = np.linalg.inv(self.Sigma_w)
              
            elif self.kitchen_kind == KitchenKind.StudentOnly:
            # ------------- The Student only Kitchen Sink ---------------- 
                self.k0, self.k1, self.k2 = COEFICIENTS['sign']

                # F = np.random.normal(0, 1, (d, p)) / np.sqrt(p)
                self.F = np.random.normal(0, 1, (self.p, self.d)) / np.sqrt(self.d)
                
                self.theta =  np.random.normal(0,1, self.p) 
                self.Sigma_w = np.eye(self.p)
                self.Sigma_w_inv = np.linalg.inv(self.Sigma_w)

                self.Psi = np.eye(self.p)
                self.Phi = self.k1 * self.F # must not be transposed! Loureiro transposes twice.
                self.Omega = np.ones(self.d).T + self.k1**2 * self.F @ self.F.T/self.d + self.k2**2 * np.eye(self.d)
            elif self.kitchen_kind == KitchenKind.DoubleCovariate:
                # ------------- The Double Covariate Kitchen Sink ----------------
                # Not technically a kitchen sink, just a covariate
                assert self.p == self.d, "p must be equal to d for the vanilla gaussian model"
                
                var = 0.1 # Choose the var too big and the state evolution will run into issues.
                self.Sigma_w = np.random.normal(0,var, (self.p,self.p)) 
                # Let's make Sigma_w positive definite
                self.Sigma_w = self.Sigma_w.T @ self.Sigma_w / self.p + np.eye(self.p)

                # self.Sigma_w = np.eye(self.p)

                self.Sigma_w_inv = np.linalg.inv(self.Sigma_w)
                
                # Let's sample the teacher weights as a normal distribution with covariance sigma_w
                self.theta = np.random.multivariate_normal(np.zeros(self.p), self.Sigma_w)

                # print the norm of theta
                logger.info("||theta|| = " + str(np.linalg.norm(self.theta)))

                # Let's go and sample Sigma_x
                self.Sigma_x = np.random.normal(0,0.1, (self.p,self.p))
                logger.info("||Sigma_x|| = " + str(np.linalg.norm(self.Sigma_x)))
                self.Sigma_x = self.Sigma_x.T @ self.Sigma_x / self.p 
                # log the norm of Sigma_x
                logger.info("||Sigma_x|| = " + str(np.linalg.norm(self.Sigma_x)))
                self.Sigma_x += np.eye(self.p)
                logger.info("||Sigma_x|| = " + str(np.linalg.norm(self.Sigma_x)))



                # Let's test if Sigma_x is positive definite
                min_eigval = np.min(np.linalg.eigvalsh(self.Sigma_x))
                if min_eigval < 0:
                    raise Exception("Sigma_x is not positive definite; min eigval: ", min_eigval)
                
                self.Psi = self.Sigma_x
                self.Omega = self.Sigma_x
                self.Phi = np.eye(self.d)

            self.gamma = self.p / self.d
            
            self.PhiPhiT = (self.Phi @ self.theta.reshape(self.p,1) @ 
                            self.theta.reshape(1,self.p) @ self.Phi.T)
            
            self.rho = self.theta.dot(self.Psi @ self.theta) / self.p

            logger.info("Sampled all data")

            # Log the shapes of PhiPhiT and Omega
            logger.info("PhiPhiT.shape = " + str(self.PhiPhiT.shape))
            logger.info("Omega.shape = " + str(self.Omega.shape))


            assumption_1 = self.Omega - self.Phi.T @ np.linalg.inv(self.Psi) @ self.Phi
            min_eigval = np.min(np.linalg.eigvalsh(assumption_1))
            if min_eigval < 0:
                raise Exception("Assumption on Schur Complement failed: Matrix was not positive semi-definite; min eigval: ", min_eigval)


            self.logger = logger


            self._check_sym()
            logger.info("Sym")
            self._diagonalise() # see base_data_model
            logger.info("Diag")
            self._check_commute()
            logger.info("commute")

            

            # pickle this object
            with open(source_pickle, 'wb') as f:
                pickle.dump(self, f)

        

    def get_info(self):
        info = {
            'data_model': 'custom',
            'teacher_dimension': self.p,
            'student_dimension': self.d,
            'aspect_ratio': self.gamma,
            'rho': self.rho
        }
        return info

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def erf(self, x):
        return 2/np.sqrt(np.pi) * np.exp(-x**2)

    def _check_sym(self):
        '''
        Check if input-input covariance is a symmetric matrix.
        '''
        if (np.linalg.norm(self.Omega - self.Omega.T) > 1e-5):
            self.logger.info('Student-Student covariance is not a symmetric matrix. Symmetrizing!')
            self.Omega = .5 * (self.Omega+self.Omega.T)

        if (np.linalg.norm(self.Psi - self.Psi.T) > 1e-5):
            self.logger.info('Teacher-teaccher covariance is not a symmetric matrix. Symmetrizing!')
            self.Psi = .5 * (self.Psi+self.Psi.T)

    def _check_commute(self):
        self.logger.info("checking commute")
        if np.linalg.norm(self.Omega @ self.PhiPhiT - self.PhiPhiT @ self.Omega) < 1e-10:
            self.commute = True
        else:
            self.commute = False
            self._UTPhiPhiTU = np.diagonal(self.eigv_Omega.T @ self.PhiPhiT @ self.eigv_Omega)

    def _diagonalise(self):
        '''
        Diagonalise covariance matrices.
        '''
        self.logger.info("diagonalising")
        self.spec_Omega, self.eigv_Omega = np.linalg.eigh(self.Omega)
        self.spec_Omega = np.real(self.spec_Omega)
        self.logger.info("Omega done...")
        self.spec_PhiPhit = np.real(np.linalg.eigvalsh(self.PhiPhiT))
        self.spec_Sigma_w_inv, self.eigv_Sigma_w_inv = np.linalg.eigh(self.Sigma_w_inv)
        self.spec_Sigma_w_inv = np.real(self.spec_Sigma_w_inv) # is it necessary to force them being real?

    def get_data(self, n, tau):
        if self.kitchen_kind == KitchenKind.StudentOnly:
            logger.info("Student only kitchen sink")
            # Student Kitchen
            c = np.random.normal(0,1,(n,self.p)) / np.sqrt(self.p)
            u = c
            y = np.sign(u @ self.theta)
            v = np.sign(1/np.sqrt(self.d) * self.F @ u.T).T
            X = v / np.sqrt(self.d)
            X_test = np.random.normal(0,1,(10000,self.p)) / np.sqrt(self.p)
            u_test = X_test
            v_test = np.sign(1/np.sqrt(self.d) * self.F @ X_test.T).T
            y_test = np.sign(1/np.sqrt(self.d) * u_test @ self.theta)
            X_test = v_test / np.sqrt(self.d)
            return X, y, X_test, y_test
            
        
        elif self.kitchen_kind == KitchenKind.TeacherStudent:

            # Teacher-Student Kitchen
            D = self.d + self.p
            c = np.random.normal(0,1,(n,D))
            u = np.sign(1/np.sqrt(D) * self.F_teacher @ c.T).T
            v = np.sign(1/np.sqrt(D) * self.F_student @ c.T).T
            y = np.sign(1/np.sqrt(self.d) * u @ self.theta)
            X = v / np.sqrt(self.d)
            X_test = np.random.normal(0,1,(10000,D)) 
            u_test = np.sign(1/np.sqrt(D) * self.F_teacher @ X_test.T).T
            v_test = np.sign(1/np.sqrt(D) * self.F_student @ X_test.T).T
            y_test = np.sign(1/np.sqrt(self.d) * u_test @ self.theta)
            X_test = v_test / np.sqrt(self.d)
            # Log the shapes
            # self.logger.info("X.shape = " + str(X.shape))
            # self.logger.info("y.shape = " + str(y.shape))
            # self.logger.info("X_test.shape = " + str(X_test.shape))
            # self.logger.info("y_test.shape = " + str(y_test.shape))
            return X, y, X_test, y_test
    
        elif self.kitchen_kind == KitchenKind.Vanilla:

            # Gaussian vanilla
            c = np.random.normal(0,1,(n,self.d)) / np.sqrt(self.d)
            u = c
            y = np.sign(1/np.sqrt(self.d) * u @ self.theta) 
            X = u
            X_test = np.random.normal(0,1,(100000,self.d)) / np.sqrt(self.d)
            y_test = np.sign(1/np.sqrt(self.d) * X_test @ self.theta)
            return X, y, X_test, y_test
        elif self.kitchen_kind == KitchenKind.DoubleCovariate:



            self.logger.info(f"Getting data for Double Covariate {n}")
            # let's sample a multivariate normal with zero mean and self.Omega covariance
            X = sample_multivariate_gaussian(np.zeros(self.p), self.Omega, n) / np.sqrt(self.p)
            # mn = multivariate_normal(mean = np.zeros(self.p), cov = self.Psi)
            # X = mn.rvs(n)/np.sqrt(self.p)
            # X = np.random.multivariate_normal(np.zeros(self.p), self.Psi, n) 
            # Log the norm of X
            self.logger.info("||X|| = " + str(np.linalg.norm(X)))
            y = np.sign(1/np.sqrt(self.p) * X @ self.theta)
            X_test = sample_multivariate_gaussian(np.zeros(self.p), self.Psi, 10000) / np.sqrt(self.p)
            # Log the norm of X_test
            self.logger.info("||X_test|| = " + str(np.linalg.norm(X_test)))
            y_test = np.sign(1/np.sqrt(self.p) * X_test @ self.theta)
            return X, y, X_test, y_test
        
            

        else:
            raise ValueError("Kitchen kind not recognised")
    


if __name__ == "__main__":
    import logging
    logger = logging.getLogger()
    # Make the logger log to console
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    
    model = RandomKitchenSinkDataModel(1000,1000, logger, delete_existing=True)
    print(model.get_info())
    # Let's store Psi in a json file
    import json
    with open("../data/Psi.json", "w") as f:
        json.dump(model.Psi.tolist(), f)