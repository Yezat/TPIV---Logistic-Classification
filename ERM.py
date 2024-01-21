"""
This module contains code for obtaining the loss, the gradient and the hessian of a given problem and an ERM estimator.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.utils.validation import check_array, check_consistent_length
from scipy.sparse.linalg import eigsh
from helpers import sigmoid,sigmoid_numba, log1pexp, log1pexp_numba, stable_cosh_squared, adversarial_loss, Task, ProblemType
from scipy.special import erfc
from data_model import *
import numba as nb

"""
------------------------------------------------------------------------------------------------------------------------
    Optimizer
------------------------------------------------------------------------------------------------------------------------
"""

def run_optimizer(task : Task, data_model: AbstractDataModel, data:DataSet, logger, df_sigma):
    
    def extract_sigma_state_evolution(dataframe, alpha, epsilon, tau, lam):
        # extract the row for the given alpha, epsilon, tau and lam
        row = dataframe.loc[alpha,epsilon,tau,lam]
        # extract the mean of sigma_state_evolution
        return float(row["mean"][0])

    epsilon = task.epsilon

    if task.problem_type == ProblemType.EquivalentLogistic:
        V = extract_sigma_state_evolution(df_sigma,task.alpha,epsilon,task.tau,task.lam)
        epsilon /= V

    w_gd = sklearn_optimize(np.random.normal(0,1,(task.d,)),data.X,data.y,task.lam,epsilon,task.problem_type, data_model.Sigma_w, data_model.Sigma_delta, logger)

    return w_gd


"""
Preprocesses the data for the sklearn optimizer.
"""
def preprocessing(coef, X, y, lam, epsilon, problem_type: ProblemType):
    # sklearn - this method expects labels as -1 and 1 and converts them to 0 and 1
    # heavily inspired by the sklearn code, with hopefully all the relevant bits copied over to make it work using lbfgs
    solver = "lbfgs"
    X = check_array(
            X,
            accept_sparse="csr",
            dtype=np.float64,
            accept_large_sparse= not solver in ["liblinear", "sag", "saga"],
        )
    y = check_array(y, ensure_2d=False, dtype=None)
    check_consistent_length(X, y)

    _, n_features = X.shape

    w0 = np.zeros(n_features, dtype=X.dtype)
    

    if coef.size not in (n_features, w0.size):
            raise ValueError(
                "Initialization coef is of shape %d, expected shape %d or %d"
                % (coef.size, n_features, w0.size)
            )
    w0[: coef.size] = coef

    if problem_type == ProblemType.Ridge:
        target = y
    elif problem_type == ProblemType.Logistic:    
        mask = y == 1
        y_bin = np.ones(y.shape, dtype=X.dtype)
        y_bin[~mask] = 0.0
        target = y_bin
    elif problem_type == ProblemType.EquivalentLogistic or ProblemType.PerturbedBoundaryLogistic:
        target = y
    else:
        raise Exception(f"Preprocessing not implemented for problem type {problem_type}")
    return w0, X, target, lam, epsilon

def sklearn_optimize(coef,X,y,lam,epsilon, problem_type: ProblemType, covariance_prior = None, sigma_delta = None, logger = None):


    w0, X,target, lam, epsilon = preprocessing(coef, X, y, lam, epsilon, problem_type)

    if covariance_prior is None:
        covariance_prior = np.eye(X.shape[1])
    if sigma_delta is None:
        sigma_delta = np.eye(X.shape[1])

    method = "L-BFGS-B"    

    loss_gd = None
    if problem_type == ProblemType.Ridge:
        problem_instance = RidgeProblem()
        loss_gd = problem_instance.loss_gradient
    elif problem_type == ProblemType.Logistic:
        problem_instance = LogisticProblem()
        loss_gd = problem_instance.loss_gradient
    elif problem_type == ProblemType.EquivalentLogistic:
        problem_instance = EquivalentLogisticProblem()
        epsilon *= lam
        loss_gd = problem_instance.loss_gradient
    elif problem_type == ProblemType.PerturbedBoundaryLogistic:
        problem_instance = PerturbedBoundaryLogisticProblem()
        loss_gd = problem_instance.loss_gradient
    else:
        raise Exception(f"Problem type {problem_type} not implemented")

    # if problem_type == ProblemType.Ridge:
    #     w0 = np.linalg.inv(X.T@X + lam * covariance_prior) @ X.T @ target
    # else:
    opt_res = minimize(
                    loss_gd,
                    w0,
                    method=method,
                    jac=True,
                    args=(X, target, lam, epsilon,covariance_prior, sigma_delta, logger),
                    options={"maxiter": 1000, "disp": False},
                )
        
    w0, _ = opt_res.x, opt_res.fun
    return w0, problem_instance


"""
------------------------------------------------------------------------------------------------------------------------
    Ridge Losses and Gradients
------------------------------------------------------------------------------------------------------------------------
"""

class RidgeProblem:

    @staticmethod
    def loss_gradient(coef, X, y, l2_reg_strength, epsilon, covariance_prior, sigma_delta, logger):
        
        loss = RidgeProblem.compute_loss(coef,X,y,l2_reg_strength,epsilon,covariance_prior,sigma_delta)

        grad = RidgeProblem.compute_gradient(coef,X,y,l2_reg_strength,epsilon,covariance_prior,sigma_delta)

        return loss, grad
    
    @staticmethod
    def compute_loss(coef, X, y, l2_reg_strength, epsilon, covariance_prior, sigma_delta):
        X = X / np.sqrt(X.shape[1])
        epsilon = epsilon / np.sqrt(X.shape[1])
        activation = X @ coef 

        wSw = coef.dot(sigma_delta@coef)
        nww = np.sqrt(coef@coef)
        adv_strength = epsilon *  wSw / nww

        loss = (y - activation + y * adv_strength).T @ (y - activation + y * adv_strength) + l2_reg_strength * (coef @ covariance_prior @ coef)

        return loss
    
    @staticmethod
    def training_loss(w,X,y,epsilon, Sigma_delta):
        return RidgeProblem.compute_loss(w,X,y,0,epsilon,np.eye(X.shape[1]),Sigma_delta)


    @staticmethod
    def compute_gradient(coef, X, y, l2_reg_strength, epsilon, covariance_prior, sigma_delta):
        X = X / np.sqrt(X.shape[1])
        activation = X.T@y

        wSw = coef.dot(sigma_delta@coef)
        nww = np.sqrt(coef@coef)
        epsilon = epsilon / np.sqrt(X.shape[1])

        adv_strength = epsilon *  wSw / nww

        XX = X.T @ X 

        YY = y.T @ y

        Delta = ((sigma_delta + sigma_delta.T) @ coef)/nww - (wSw / nww**3) * coef

        grad = -2*activation+ 2*XX@coef + 2 * epsilon * YY * Delta
        grad += -2*activation*adv_strength - 2*epsilon* coef.T @ activation * Delta
        grad += 2*epsilon * YY * adv_strength*Delta

        # regularization
        grad += l2_reg_strength * (covariance_prior + covariance_prior.T) @ coef

        return grad



"""
------------------------------------------------------------------------------------------------------------------------
    Logistic Losses and Gradients
------------------------------------------------------------------------------------------------------------------------
"""

class LogisticProblem:

    @staticmethod
    def loss_gradient(coef, X, y,l2_reg_strength, epsilon, covariance_prior, sigma_delta, logger):
        n_features = X.shape[1]
        weights = coef
        raw_prediction = X @ weights / np.sqrt(n_features)    

        l2_reg_strength /= 2

        wSw = weights.dot(sigma_delta@weights)
        nww = np.sqrt(weights@weights)

        optimal_attack = epsilon/np.sqrt(n_features) *  wSw / nww 

        loss = LogisticProblem.compute_loss(raw_prediction,optimal_attack,y)
        loss = loss.sum()
        loss +=  l2_reg_strength * (weights @ covariance_prior @ weights)


        epsilon_gradient_per_sample,gradient_per_sample = LogisticProblem.compute_gradient(raw_prediction,optimal_attack,y)   

        derivative_optimal_attack = epsilon/np.sqrt(n_features) * ( 2*sigma_delta@weights / nww  - ( wSw / nww**3 ) * weights )

        adv_grad_summand = np.outer(epsilon_gradient_per_sample, derivative_optimal_attack).sum(axis=0)


        # if epsilon is zero, assert that the norm of adv_grad_summand is zero
        if epsilon == 0:
            assert np.linalg.norm(adv_grad_summand) == 0, f"derivative_optimal_attack {np.linalg.norm(derivative_optimal_attack)}, epsilon_gradient_per_sample {np.linalg.norm(epsilon_gradient_per_sample)}"



        grad = np.empty_like(coef, dtype=weights.dtype)
        grad[:n_features] = X.T @ gradient_per_sample / np.sqrt(n_features) +  l2_reg_strength * ( covariance_prior + covariance_prior.T) @ weights + adv_grad_summand

        return loss, grad

    """
    ------------------------------------------------------------------------------------------------------------------------
        Loss
    ------------------------------------------------------------------------------------------------------------------------
    """

    @staticmethod
    def compute_loss(z,e,y):
        return -y*z + y*e + (1-y)*log1pexp(z+e) + y*log1pexp(z-e)

    @staticmethod
    def training_loss_with_regularization(w,X,y,lam,epsilon,covariance_prior = None):
        z = X@w
        if covariance_prior is None:
            covariance_prior = np.eye(X.shape[1])
        return (adversarial_loss(y,z,epsilon/np.sqrt(X.shape[1]),w@w).sum() + 0.5 * lam * w@covariance_prior@w )/X.shape[0]

    @staticmethod
    def training_loss(w,X,y,epsilon, Sigma_delta):
        z = X@w/np.sqrt(X.shape[1])
        attack = epsilon/np.sqrt(X.shape[1]) * ( w.dot(Sigma_delta@w) / np.sqrt(w@w)  )
        return (adversarial_loss(y,z,attack).sum())/X.shape[0]


    """
    ------------------------------------------------------------------------------------------------------------------------
        Gradient
    ------------------------------------------------------------------------------------------------------------------------
    """

    @staticmethod
    def compute_gradient(z,e,y):
        opt_attack_term = (1-y)*sigmoid(z+e) + y*sigmoid(-z+e)
        data_term = (1-y)*sigmoid(z+e) - y*sigmoid(-z+e)
        return opt_attack_term, data_term


    """
    ------------------------------------------------------------------------------------------------------------------------
        Hessian
    ------------------------------------------------------------------------------------------------------------------------
    """

    @staticmethod
    def compute_hessian(X,y,theta,epsilon, lam, Sigma_w):
        X = X / np.sqrt(X.shape[1])
        raw_prediction = X.dot(theta)

        # B - Optimal Attack ()
        B = epsilon * np.linalg.norm(theta) / np.sqrt(X.shape[1])

        # C and C_prime (n,)
        C = raw_prediction + B
        C_prime = raw_prediction - B

        # H - Derivative of Optimal Attack (d,)
        H = epsilon * theta / (np.linalg.norm(theta) * np.sqrt(X.shape[1]))

        # dH - Hessian of Optimal Attack (d,d)
        dH = np.eye(X.shape[1]) * epsilon / (np.linalg.norm(theta) * np.sqrt(X.shape[1])) - epsilon*np.outer(theta, theta) / (np.linalg.norm(theta) ** 3 * np.sqrt(X.shape[1]))

        # dH term
        vec = (1-y) * sigmoid(C) + y * sigmoid(-C_prime) # (n,)
        hessian = vec.sum() * dH

        # dC term and dC_prime term
        vecC = (1-y) * stable_cosh_squared(C) # (n,)
        vecC_prime = y * stable_cosh_squared(C_prime) # (n,)

        # Shift X by H
        X_plus = X + H
        X_minus = X - H

        # dC term
        act = np.multiply(X_plus.T, vecC) # (d,n)
        hessian += np.einsum('ij,ik->jk', X_plus, act.T) # (d,d)

        # dC_prime term
        act = np.multiply(X_minus.T, vecC_prime) # (d,n)
        hessian += np.einsum('ij,ik->jk', X_minus, act.T) # (d,d)

        # Regularization
        hessian += lam/2 * (Sigma_w + Sigma_w.T)

        return hessian

    @staticmethod
    def min_eigenvalue_hessian(X,y,theta,epsilon, lam, Sigma_w):
        hessian = LogisticProblem.compute_hessian(X,y,theta,epsilon, lam, Sigma_w)
        # return np.min(eigvalsh(hessian))
        return eigsh(hessian, k=1, which='SA')[0][0]


"""
------------------------------------------------------------------------------------------------------------------------
    Logistic Losses and Gradients
------------------------------------------------------------------------------------------------------------------------
"""

class EquivalentLogisticProblem:

    @staticmethod
    def loss_gradient(coef, X, y,l2_reg_strength, epsilon, covariance_prior, sigma_delta, logger):
        n_features = X.shape[1]
        weights = coef
        raw_prediction = X @ weights / np.sqrt(n_features)    

        l2_reg_strength /= 2

        wSw = weights.dot(sigma_delta@weights)
        nww = np.sqrt(weights@weights)

        optimal_attack = epsilon/np.sqrt(n_features) *  wSw / nww 

        loss = EquivalentLogisticProblem.compute_loss(raw_prediction,optimal_attack,y)
        loss = loss.sum()
        loss +=  l2_reg_strength * (weights @ covariance_prior @ weights)


        gradient_per_sample = EquivalentLogisticProblem.compute_gradient(raw_prediction,optimal_attack,y)   


        adv_grad_summand = epsilon*np.sum(X * y / np.sqrt(n_features),axis=1)
        assert adv_grad_summand.shape == (n_features,), f"adv_grad_summand.shape = {adv_grad_summand.shape}"


        # if epsilon is zero, assert that the norm of adv_grad_summand is zero
        if epsilon == 0:
            assert np.linalg.norm(adv_grad_summand) == 0    
                

        
        grad = np.empty_like(coef, dtype=weights.dtype)
        grad[:n_features] = X.T @ gradient_per_sample / np.sqrt(n_features) +  l2_reg_strength * ( covariance_prior + covariance_prior.T) @ weights + adv_grad_summand

        return loss, grad

    """
    ------------------------------------------------------------------------------------------------------------------------
        Loss
    ------------------------------------------------------------------------------------------------------------------------
    """

    @staticmethod
    def compute_loss(z,e,y):
        return log1pexp(-y*z) + e*y*z

    @staticmethod
    def training_loss_with_regularization(w,X,y,lam,epsilon,covariance_prior = None):
        z = X@w
        if covariance_prior is None:
            covariance_prior = np.eye(X.shape[1])
        return (adversarial_loss(y,z,epsilon/np.sqrt(X.shape[1]),w@w).sum() + 0.5 * lam * w@covariance_prior@w )/X.shape[0]

    @staticmethod
    def training_loss(w,X,y,epsilon, Sigma_delta):
        z = X@w/np.sqrt(X.shape[1])
        attack = epsilon/np.sqrt(X.shape[1]) * ( w.dot(Sigma_delta@w) / np.sqrt(w@w)  )
        return (adversarial_loss(y,z,attack).sum())/X.shape[0]


    """
    ------------------------------------------------------------------------------------------------------------------------
        Gradient
    ------------------------------------------------------------------------------------------------------------------------
    """

    @staticmethod
    def compute_gradient(z,e,y):
        gradient_per_sample = sigmoid(-y*z) 
        return gradient_per_sample

"""
------------------------------------------------------------------------------------------------------------------------
    Perturbed Boundary Logistic Problem
------------------------------------------------------------------------------------------------------------------------
"""

class PerturbedBoundaryLogisticProblem():

    @staticmethod
    def loss_gradient(coef, X, y,l2_reg_strength, epsilon, covariance_prior, sigma_delta, logger):
        n_features = X.shape[1]
        weights = coef
        raw_prediction = X @ weights / np.sqrt(n_features)    

        l2_reg_strength /= 2

        wSw = weights.dot(sigma_delta@weights)
        nww = np.sqrt(weights@weights)

        optimal_attack = epsilon/np.sqrt(n_features) *  wSw / nww 

        shifted_margins = y*raw_prediction - optimal_attack

        # mask the shifted margins where they are positive
        mask_positive = shifted_margins > 0

        # compute corresponding subsets
        shifted_margins_positive = shifted_margins[mask_positive]
        shifted_margins_negative = shifted_margins[~mask_positive]


        loss = PerturbedBoundaryLogisticProblem.compute_loss(shifted_margins_positive, shifted_margins_negative)
        loss +=  l2_reg_strength * (weights @ covariance_prior @ weights)


        positive_gradient_per_sample, negative_gradient_per_sample = PerturbedBoundaryLogisticProblem.compute_gradient(shifted_margins_positive, shifted_margins_negative)   

        derivative_optimal_attack = epsilon/np.sqrt(n_features) * ( 2*sigma_delta@weights / nww  - ( wSw / nww**3 ) * weights )

        positive_adv_grad_summand = np.outer(positive_gradient_per_sample, derivative_optimal_attack).sum(axis=0)
        negative_adv_grad_summand = np.outer(negative_gradient_per_sample, derivative_optimal_attack).sum(axis=0)


        # if epsilon is zero, assert that the norm of adv_grad_summand is zero
        if epsilon == 0:
            assert np.linalg.norm(positive_adv_grad_summand) == 0, f"derivative_optimal_attack {np.linalg.norm(derivative_optimal_attack)}, gradient_per_sample {np.linalg.norm(positive_gradient_per_sample)}"
            assert np.linalg.norm(negative_adv_grad_summand) == 0, f"derivative_optimal_attack {np.linalg.norm(derivative_optimal_attack)}, gradient_per_sample {np.linalg.norm(negative_gradient_per_sample)}"


        positive_data = X[mask_positive]
        negative_data = X[~mask_positive]

        positive_labels = y[mask_positive]
        negative_labels = y[~mask_positive]

        positive_label_data_product = positive_labels[:,np.newaxis] * positive_data / np.sqrt(n_features)
        positive_contribution= positive_label_data_product.T @ positive_gradient_per_sample
        # log the shape
        # logger.info(f"positive_contribution.shape = {positive_contribution.shape}")

        negative_label_data_product = negative_labels[:,np.newaxis] * negative_data / np.sqrt(n_features)
        negative_contribution= negative_label_data_product.T @ negative_gradient_per_sample
        # log the shape
        # logger.info(f"negative_contribution.shape = {negative_contribution.shape}")

        grad = np.empty_like(coef, dtype=weights.dtype)
        grad[:n_features] = positive_contribution + negative_contribution +  l2_reg_strength * ( covariance_prior + covariance_prior.T) @ weights + positive_adv_grad_summand + negative_adv_grad_summand

        return loss, grad

    """
    ------------------------------------------------------------------------------------------------------------------------
        Loss
    ------------------------------------------------------------------------------------------------------------------------
    """

    @staticmethod
    def compute_loss(shifted_margins_positive, shifted_margins_negative):
        return np.sum(log1pexp(-shifted_margins_positive)) + np.sum( np.log(2) -0.5*shifted_margins_negative + (1/8)*shifted_margins_negative**2)

    @staticmethod
    def training_loss_with_regularization(w,X,y,lam,epsilon,covariance_prior = None):
        z = X@w
        if covariance_prior is None:
            covariance_prior = np.eye(X.shape[1])
        return (adversarial_loss(y,z,epsilon/np.sqrt(X.shape[1]),w@w).sum() + 0.5 * lam * w@covariance_prior@w )/X.shape[0]

    @staticmethod
    def training_loss(w,X,y,epsilon, Sigma_delta):
        z = X@w/np.sqrt(X.shape[1])
        attack = epsilon/np.sqrt(X.shape[1]) * ( w.dot(Sigma_delta@w) / np.sqrt(w@w)  )
        return (adversarial_loss(y,z,attack).sum())/X.shape[0]


    """
    ------------------------------------------------------------------------------------------------------------------------
        Gradient
    ------------------------------------------------------------------------------------------------------------------------
    """

    @staticmethod
    def compute_gradient(shifted_margins_positive, shifted_margins_negative):
        positive_part = -sigmoid(-shifted_margins_positive)
        negative_part = -0.5 + shifted_margins_negative/4
        return positive_part, negative_part


"""
------------------------------------------------------------------------------------------------------------------------
    Errors
------------------------------------------------------------------------------------------------------------------------
"""
def error(y, yhat):
    return 0.25*np.mean((y-yhat)**2)

def adversarial_error(y, Xtest, w_gd, epsilon, Sigma_upsilon):
    d = Xtest.shape[1]
    wSw = w_gd.dot(Sigma_upsilon@w_gd)
    nww = np.sqrt(w_gd@w_gd)

    return error(y, np.sign( Xtest@w_gd/np.sqrt(d) - y*epsilon/np.sqrt(d) * wSw/nww  ))

def compute_boundary_loss(y, Xtest, epsilon, sigma_delta, w_gd, l2_reg_strength):
    d = Xtest.shape[1]
    wSw = w_gd.dot(sigma_delta@w_gd)
    nww = np.sqrt(w_gd@w_gd)

    optimal_attack = epsilon/np.sqrt(d) *  wSw / nww

    raw_prediction = Xtest @ w_gd / np.sqrt(d)

    # log shape of raw_prediction
    # logger.info(f"raw_prediction.shape = {raw_prediction.shape}")

    # compute y * raw_prediction elementwise and sum over all samples
    y_raw_prediction = y * raw_prediction
    y_raw_prediction_sum = y_raw_prediction.sum()

    # log y_raw_prediction_sum shape
    # logger.info(f"y_raw_prediction_sum.shape = {y_raw_prediction_sum.shape}")


    boundary_loss = y_raw_prediction_sum*optimal_attack*l2_reg_strength

    # # log boundary_loss shape
    # logger.info(f"boundary_loss.shape = {boundary_loss.shape}")

    # assert boundary_loss to be a scalar
    assert np.isscalar(boundary_loss)

    return boundary_loss

def adversarial_error_teacher(y, Xtest, w_gd, teacher_weights, epsilon, data_model):
    if teacher_weights is None:
        return None

    d = Xtest.shape[1]
    
    nww = np.sqrt(w_gd@w_gd)

    tSw = teacher_weights.dot(data_model.Sigma_upsilon@w_gd) # shape (d,)
    
    y_attacked_teacher = np.sign( Xtest@teacher_weights/np.sqrt(d) - y*epsilon/np.sqrt(d) * tSw/nww  )

    return error(y_attacked_teacher, y)

def fair_adversarial_error_erm(X_test, w_gd, teacher_weights, epsilon, gamma, data_model, logger = None):
    
    d = X_test.shape[1]

    N = w_gd@w_gd 
    A = w_gd.dot(data_model.Sigma_upsilon@w_gd) 
    F = w_gd.dot(data_model.Sigma_upsilon@teacher_weights)
    teacher_activation = X_test@teacher_weights/np.sqrt(d)
    student_activation = X_test@w_gd/np.sqrt(d)
 
    y = np.sign(teacher_activation)

    gamma_constraint_argument = y*teacher_activation - epsilon*F/np.sqrt(N*d)

    # first term    
    y_first = np.zeros_like(y)
    y_gamma = np.zeros_like(y)
    moved_argument = student_activation + A/F * (y* gamma - teacher_activation )
    y_gamma_t = np.sign( moved_argument )
    mask_gamma_smaller = (y*teacher_activation < gamma + epsilon*F/np.sqrt(N*d)) & (y*teacher_activation > gamma)
    y_gamma[mask_gamma_smaller] = y_gamma_t[mask_gamma_smaller]
    y_first[mask_gamma_smaller] = y[mask_gamma_smaller]    
    first_error = error(y_first, y_gamma)


    
    # second term
    y_second = np.zeros_like(y)
    y_max = np.zeros_like(y)
    mask_gamma_bigger = (gamma_constraint_argument >= gamma) & (y*teacher_activation > gamma)    
    y_max_t = np.sign( student_activation - y * epsilon * A/np.sqrt(N*d) )
    y_max[mask_gamma_bigger]  = y_max_t[mask_gamma_bigger]
    y_second[mask_gamma_bigger] = y[mask_gamma_bigger]
    second_error = error(y_second, y_max)


    # third term
    y_hat = np.zeros_like(y)
    y_third = np.zeros_like(y)
    mask_last_smaller = (y*teacher_activation <= gamma) & (y * teacher_activation > 0)
    y_hat_t = np.sign( X_test@w_gd )    
    y_hat[mask_last_smaller] = y_hat_t[mask_last_smaller]
    y_third[mask_last_smaller] = y[mask_last_smaller]
    third_error = error(y_third, y_hat)
        


    return first_error + second_error + third_error
    



"""
------------------------------------------------------------------------------------------------------------------------
    Calibration
------------------------------------------------------------------------------------------------------------------------
"""
def compute_experimental_teacher_calibration(p, w, werm, Xtest, sigma):
    try:

        
        # size of bins where we put the probas
        n, d = Xtest.shape
        dp = 0.025
        
        def probit(lf, sigma):
            return 0.5 * erfc(- lf / np.sqrt(2 * sigma**2))
        

        Ypred = sigmoid(Xtest @ werm / np.sqrt(d))
        

        index = [i for i in range(n) if p - dp <= Ypred[i] <= p + dp]
        

        if sigma == 0:
            teacher_probabilities = np.array([np.heaviside(Xtest[i] @ w / np.sqrt(d),0.5) for i in index]) 
        else:
            teacher_probabilities = np.array([probit(Xtest[i] @ w / np.sqrt(d),sigma) for i in index]) 

        if len(teacher_probabilities) == 0:
            return p-1

        return p - np.mean(teacher_probabilities)
    

    except Exception as e:
        # probably a mean of empty slice... is it an exception though?
        print(e)
        return np.nan
    
"""
------------------------------------------------------------------------------------------------------------------------
    Predictions
------------------------------------------------------------------------------------------------------------------------
"""
def predict_erm(X,weights):
    return np.sign(predict_erm_probability(X,weights) - 0.5)

def predict_erm_probability(X,weights):
    argument = X@weights/np.sqrt(X.shape[1])
    
    return sigmoid(argument)

