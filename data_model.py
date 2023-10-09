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

        self.d = d
        # Psi 
        self.Psi = np.eye(d)
        self.Omega = np.eye(d)
        # Phi is a zero matrix
        self.Phi = np.eye(d)
        self.theta = np.random.normal(0,1, d) 

        self.Sigma_w = np.eye(d)
        
        self.p, self.k = self.Phi.shape
        self.gamma = self.k / self.p
        
        # What if we fixed self.theta # self.theta.T?
        

        self.PhiPhiT = (1 * self.Phi @ self.Phi.T)
        
        self.rho = self.theta.dot(self.Psi @ self.theta) / self.k
        self.rho = 1

        self._diagonalise() # see base_data_model (should not be necessary)
        self._check_commute()


    def get_data(self, n, tau):
        
        # theta = sample_weights(self.d)
        # Xtrain, y = sample_training_data(theta,self.d,n,tau)
        # n_test = 100000
        # Xtest,ytest = sample_training_data(theta,self.d,n_test,tau)
        # return Xtrain, y, Xtest, ytest, theta

        the = np.random.normal(0,1, self.d) 
        X = np.random.normal(0,1,(n,self.d)) / np.sqrt(self.d)
        y = np.sign(1/np.sqrt(self.d) * X @ the + tau * np.random.normal(0,1,(n,)) )
        n_test = 100000
        X_test = np.random.normal(0,1,(n_test,self.d)) / np.sqrt(self.d)
        y_test = np.sign(1/np.sqrt(self.d) * X_test @ the + tau * np.random.normal(0,1,(n_test,)))
        return X, y, X_test, y_test, the
    
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


class KitchenKind(Enum):
    Vanilla = 1
    StudentOnly = 2
    TeacherStudent = 3
    DoubleCovariate = 4
    VanillaStrongWeak = 5 # only some eigenvalues are stronger than others...
    StrongWeak = 6 # If there are covariances
    SourceCapacity = 7





class RandomKitchenSinkDataModel(DataModel):
    def __init__(self, student_dimension, teacher_dimension, logger, source_pickle_path = "../", delete_existing = False):  

        self.logger = logger
 
        logger.info("Let that Random Kitchen Sink in")

        self.kitchen_kind = KitchenKind.SourceCapacity

        self.d = student_dimension
        self.p = teacher_dimension
        self.gamma = self.p / self.d



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

                self.rho = 1
                self.PhiPhiT = np.eye(self.d)      
                self.Sigma_x = np.eye(self.p)         
                self.Sigma_delta = np.eye(self.p) 

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
              
            elif self.kitchen_kind == KitchenKind.StudentOnly:
            # ------------- The Student only Kitchen Sink ---------------- 
                self.k0, self.k1, self.k2 = COEFICIENTS['sign']

                # F = np.random.normal(0, 1, (d, p)) / np.sqrt(p)
                self.F = np.random.normal(0, 1, (self.p, self.d)) / np.sqrt(self.d)
                
                self.theta =  np.random.normal(0,1, self.p) 
                self.Sigma_w = np.eye(self.p)

                self.Psi = np.eye(self.p)
                self.Phi = self.k1 * self.F # must not be transposed! Loureiro transposes twice.
                self.Omega = np.ones(self.d).T + self.k1**2 * self.F @ self.F.T/self.d + self.k2**2 * np.eye(self.d)
            elif self.kitchen_kind == KitchenKind.DoubleCovariate:
                # ------------- The Double Covariate Kitchen Sink ----------------
                # Not technically a kitchen sink, just a covariate
                assert self.p == self.d, "p must be equal to d for the vanilla gaussian model"
                
                var = 0.5 # Choose the var too big and the state evolution will run into issues.
                self.Sigma_w = np.random.normal(0,var, (self.p,self.p)) 
                # Let's make Sigma_w positive definite
                self.Sigma_w = self.Sigma_w.T @ self.Sigma_w / self.p + np.eye(self.p)
                
                # Let's sample the teacher weights as a normal distribution with covariance sigma_w
                self.theta = np.random.multivariate_normal(np.zeros(self.p), self.Sigma_w)


                # Let's go and sample Sigma_x
                self.Sigma_x = np.random.normal(0,0.9, (self.p,self.p))
                self.Sigma_x = self.Sigma_x.T @ self.Sigma_x / self.p
                self.Sigma_x += 1 *  np.eye(self.p)



                # Let's test if Sigma_x is positive definite
                min_eigval = np.min(np.linalg.eigvalsh(self.Sigma_x))
                if min_eigval < 0:
                    raise Exception("Sigma_x is not positive definite; min eigval: ", min_eigval)
                
                self.Psi = self.Sigma_x
                self.Omega = self.Sigma_x
                self.Phi = self.Sigma_x

                self.PhiPhiT = self.Phi @ self.Sigma_w @ self.Phi.T
                
                # Let's get the eigenvalues of Psi
                self.spec_Psi = np.linalg.eigvalsh(self.Psi)
                # Let's get the eigenvalues of Sigma_w
                self.spec_Sigma_w = np.linalg.eigvalsh(self.Sigma_w)
                self.rho = self.spec_Psi.dot(self.spec_Sigma_w) / self.p

                self.Sigma_delta = np.eye(self.p)

            elif self.kitchen_kind == KitchenKind.VanillaStrongWeak:
                # ------------------ The Vanilla Identity Gaussian Model with Stronger Eigenvalues ----------------
                assert self.p == self.d, "p must be equal to d for the vanilla strong-weak model"

                # TODO Make sure to distinguish teacher and student covariances for the weight priors

                # For now we will just increase the first few eigenvalues
                self.Sigma_w = np.eye(self.p)
                d1 = 1
                # self.Sigma_w[:d1,:d1] = 1 * np.eye(d1)
                self.Sigma_w[0,0] = 100

                

                # Sample the teacher from Sigma_w
                self.theta = np.random.multivariate_normal(np.zeros(self.p), self.Sigma_w)

                # Let's for now take Sigma_x = Sigma_w
                self.Sigma_x = np.eye(self.p)

                self.Sigma_x[0,0] = 100

                self.Psi = self.Sigma_x
                self.Omega = self.Sigma_x
                self.Phi = self.Sigma_x

                self.PhiPhiT = self.Phi @ self.Sigma_w @ self.Phi.T
                
                # Let's get the eigenvalues of Psi
                self.spec_Psi = np.linalg.eigvalsh(self.Psi)
                # Let's get the eigenvalues of Sigma_w
                self.spec_Sigma_w = np.linalg.eigvalsh(self.Sigma_w)
                self.rho = self.spec_Psi.dot(self.spec_Sigma_w) / self.p

                self.Sigma_delta = np.eye(self.p)
            elif self.kitchen_kind == KitchenKind.SourceCapacity:
                alph = 1.2
                r = 0.3

                

                spec_Omega0 = np.array([self.p/(k+1)**alph for k in range(self.p)])
                self.Omega=np.diag(spec_Omega0)
                self.Phi = self.Omega
                self.Psi = self.Omega

                self.theta = np.sqrt(np.array([1/(k+1)**((1+alph*(2*r-1))) for k in range(self.p)]))
                self.rho = np.mean(spec_Omega0 * self.theta**2) 
                self.PhiPhiT = np.diag(spec_Omega0**2 * self.theta**2)

                self.Sigma_delta = np.eye(self.p)
                

                self.Sigma_t = np.diag(self.theta**2)
                self.Sigma_w  = np.eye(self.p)

            
            
            # self.PhiPhiT = (self.Phi @ self.theta.reshape(self.p,1) @ 
            #                 self.theta.reshape(1,self.p) @ self.Phi.T)
            
            
            # self.rho = self.theta.dot(self.Psi @ self.theta) / self.p

            logger.info("Sampled all data")

            # Log the shapes of PhiPhiT and Omega
            logger.info("PhiPhiT.shape = " + str(self.PhiPhiT.shape))
            logger.info("Omega.shape = " + str(self.Omega.shape))


            assumption_1 = self.Omega - self.Phi.T @ np.linalg.inv(self.Psi) @ self.Phi
            min_eigval = np.min(np.linalg.eigvalsh(assumption_1))
            if min_eigval < 0:
                logger.warning(f"Assumption on Schur Complement failed: Matrix was not positive semi-definite; min eigval: {min_eigval}")


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
            'rho': self.rho,
            'commute': self.commute,
            'kitchen_kind': self.kitchen_kind.name,
            'Norm Sigma_w': np.linalg.norm(self.Sigma_w),
            'Norm Psi': np.linalg.norm(self.Psi),
            'Norm Omega': np.linalg.norm(self.Omega),
            'Norm Phi': np.linalg.norm(self.Phi),
            'Norm PhiPhiT': np.linalg.norm(self.PhiPhiT),
            'Norm theta': np.linalg.norm(self.theta),
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
        self.spec_Sigma_w = np.real(np.linalg.eigvalsh(self.Sigma_w))
        self.spec_Sigma_delta = np.real(np.linalg.eigvalsh(self.Sigma_delta))

    def get_data(self, n, tau):
        if self.kitchen_kind == KitchenKind.StudentOnly:
            logger.info("Student only kitchen sink")
            # Student Kitchen
            c = np.random.normal(0,1,(n,self.p)) / np.sqrt(self.p)
            u = c
            y = np.sign(u @ self.theta)
            v = np.sign(1/np.sqrt(self.d) * self.F @ u.T + tau * np.random.normal(0,1,(n,))).T
            X = v / np.sqrt(self.d)
            n_test = 10000
            X_test = np.random.normal(0,1,(n_test,self.p)) / np.sqrt(self.p)
            u_test = X_test
            v_test = np.sign(1/np.sqrt(self.d) * self.F @ X_test.T).T
            y_test = np.sign(1/np.sqrt(self.d) * u_test @ self.theta + + tau * np.random.normal(0,1,(n_test,)))
            X_test = v_test / np.sqrt(self.d)
            return X, y, X_test, y_test
            
        
        elif self.kitchen_kind == KitchenKind.TeacherStudent:

            # Teacher-Student Kitchen
            D = self.d + self.p
            c = np.random.normal(0,1,(n,D))
            u = np.sign(1/np.sqrt(D) * self.F_teacher @ c.T).T
            v = np.sign(1/np.sqrt(D) * self.F_student @ c.T).T
            y = np.sign(1/np.sqrt(self.d) * u @ self.theta + tau * np.random.normal(0,1,(n,)))
            X = v / np.sqrt(self.d)
            n_test = 10000
            X_test = np.random.normal(0,1,(n_test,D)) 
            u_test = np.sign(1/np.sqrt(D) * self.F_teacher @ X_test.T).T
            v_test = np.sign(1/np.sqrt(D) * self.F_student @ X_test.T).T
            y_test = np.sign(1/np.sqrt(self.d) * u_test @ self.theta + tau * np.random.normal(0,1,(n_test,)))
            X_test = v_test / np.sqrt(self.d)
            # Log the shapes
            # self.logger.info("X.shape = " + str(X.shape))
            # self.logger.info("y.shape = " + str(y.shape))
            # self.logger.info("X_test.shape = " + str(X_test.shape))
            # self.logger.info("y_test.shape = " + str(y_test.shape))
            return X, y, X_test, y_test
    
        elif self.kitchen_kind == KitchenKind.Vanilla:
            the = np.random.normal(0,1, self.d)
            # Gaussian vanilla
            c = np.random.normal(0,1,(n,self.d)) / np.sqrt(self.d)
            u = c
            y = np.sign(1/np.sqrt(self.d) * u @ the + tau * np.random.normal(0,1,(n,))) 
            X = u
            X_test = np.random.normal(0,1,(100000,self.d)) / np.sqrt(self.d)
            y_test = np.sign(1/np.sqrt(self.d) * X_test @ the + tau * np.random.normal(0,1,(100000,)))
            return X, y, X_test, y_test, the
        elif self.kitchen_kind == KitchenKind.DoubleCovariate or self.kitchen_kind == KitchenKind.VanillaStrongWeak or self.kitchen_kind == KitchenKind.StrongWeak:


            self.logger.info(f"Getting data for {self.kitchen_kind.name} {n}")

            # the = sample_multivariate_gaussian(np.zeros(self.p), self.Sigma_w,1
            the = np.random.default_rng().multivariate_normal(np.zeros(self.p), self.Sigma_w, method="cholesky")

            # let's sample a multivariate normal with zero mean and self.Omega covariance (both scipy and np are extremely slow in mps on my machine) depends on the method I believe.. and maybe some locking issues
            
            X = np.random.default_rng().multivariate_normal(np.zeros(self.p), self.Psi, n, method="cholesky") / np.sqrt(self.p)            
            y = np.sign(X @ the + tau * np.random.normal(0,1,(n,)))
            
            X_test = np.random.default_rng().multivariate_normal(np.zeros(self.p), self.Psi, 10000, method="cholesky") / np.sqrt(self.p)
            # Log the norm of X_test
            
            y_test = np.sign(X_test @ the + tau * np.random.normal(0,1,(10000,)))
            return X, y, X_test, y_test, the
        elif self.kitchen_kind == KitchenKind.SourceCapacity:
            self.logger.info(f"Getting data for {self.kitchen_kind.name} {n}")

            # let's sample a multivariate normal with zero mean and self.Omega covariance (both scipy and np are extremely slow in mps on my machine) depends on the method I believe.. and maybe some locking issues
            X = np.random.default_rng().multivariate_normal(np.zeros(self.p), self.Psi, n, method="cholesky")  / np.sqrt(self.p)
            y = np.sign(X @ self.theta + tau * np.random.normal(0,1,(n,)))
            
            X_test = np.random.default_rng().multivariate_normal(np.zeros(self.p), self.Psi, 10000, method="cholesky") / np.sqrt(self.p)
            y_test = np.sign(X_test @ self.theta + tau * np.random.normal(0,1,(10000,)))

            return X, y, X_test, y_test, self.theta
        
            

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


    # print the shapes of spec_Omega and spec_PhiPhit
    print(model.spec_Omega.shape)
    print(model.spec_PhiPhit.shape)

    # count the unique values of spec_Omega
    # print(np.unique(model.spec_Omega, return_counts=True))
    # print(np.unique(model.spec_PhiPhit, return_counts=True))

 
    # Get the unique values of the eigenspectru
    # print(np.unique(np.linalg.eigvalsh(model.Psi), return_counts=True))

    # Let's sample some data
    # X, y, X_test, y_test, the = model.get_data(10000, 0.1)

