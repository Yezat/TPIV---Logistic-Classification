from gradient_descent import *
from sklearn._loss import HalfBinomialLoss
from sklearn.linear_model._linear_loss import LinearModelLoss

import sklearn_loss as skloss



# just_loss_gradient
# c_inspired_loss_gradient

# loss = LinearModelLoss(
#                 base_loss=HalfBinomialLoss(), fit_intercept=False
#             )
    # func = loss.loss_gradient
    # func = c_inspired_loss_gradient 
# coef, X, y, sample_weight=None, l2_reg_strength=0.0, n_threads=1

if __name__ == "__main__":
    d = 500
    alpha = 0.3
    tau = 2
    for lam in np.array([0,1e-4,1e-3,1e-2,1e-1,1]):
        w = sample_weights(d)
        Xtrain, y = sample_training_data(w,d,int(alpha * d),tau)

        loss = LinearModelLoss(
                base_loss=HalfBinomialLoss(), fit_intercept=False
            )
        func = loss.loss_gradient

        

        sk_loss, sk_gradient = func(w,Xtrain,y,None,lam,1)
        my_loss, my_gradient = c_inspired_loss_gradient(w,Xtrain,y,None,lam,1)
        # print
        print(f"lam={lam}, sk_loss={sk_loss}, my_loss={my_loss}")


        raw_prediction = Xtrain @ w
        sample_weight = None
        n_threads = 1
        sk_loss, sk_gradient = loss.base_loss.loss_gradient(
            y_true=y,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            n_threads=n_threads,
        )
        my_loss, my_gradient = just_loss_gradient(y,raw_prediction)
        # print
        print(f"lam={lam}, sk_loss={sk_loss.sum()}, my_loss={my_loss.sum()}")

        half = skloss.CyHalfBinomialLoss()

        loss_out = None
        gradient_out = None
        if loss_out is None:
            if gradient_out is None:
                loss_out = np.empty_like(y)
                gradient_out = np.empty_like(raw_prediction)
            else:
                loss_out = np.empty_like(y, dtype=gradient_out.dtype)
        elif gradient_out is None:
            gradient_out = np.empty_like(raw_prediction, dtype=loss_out.dtype)

        # Be graceful to shape (n_samples, 1) -> (n_samples,)
        if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
            raw_prediction = raw_prediction.squeeze(1)
        if gradient_out.ndim == 2 and gradient_out.shape[1] == 1:
            gradient_out = gradient_out.squeeze(1)

        half.loss_gradient(             y_true=y,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            loss_out=loss_out,
            gradient_out=gradient_out,
            n_threads=n_threads,
        )
        l,g = loss_out,gradient_out
        # print
        print(f"lam={lam}, sk_loss={l.sum()}, my_loss={my_loss.sum()}")
