import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.optimize import minimize

# Helper function for local linear regression as a part of the Super Learner
def local_linear_fit(X, Y, new_X, predict_X=None, bandwidth=1.0):
    """
    Fit local linear regression model and predict using new data.
    
    Parameters:
    - X: array-like of shape (n_samples,)
        Independent variable data.
    - Y: array-like of shape (n_samples,)
        Dependent variable data.
    - new_X: array-like
        Points at which the model is fitted (centers of the local models).
    - predict_X: array-like, optional
        Points at which predictions are desired. If None, predictions are made at new_X.
    - bandwidth: float, default=1.0
        Bandwidth parameter for the Gaussian kernel.
    
    Returns:
    - predictions: array of shape (len(predict_X) or len(new_X),)
        Predicted values.
    """
    n = len(X)
    predictions = []

    # If predict_X is not provided, predict at new_X
    if predict_X is None:
        predict_X = new_X

    # Apply local linear regression using a Gaussian kernel
    for x0 in new_X:
        # Compute weights based on distance from x0
        weights = np.exp(-((X - x0)**2) / (2 * bandwidth**2))
        W = np.diag(weights)
        X_aug = np.vstack([np.ones(n), X - x0]).T
        beta = np.linalg.inv(X_aug.T @ W @ X_aug) @ (X_aug.T @ W @ Y)
        
        # Predict at points relative to x0
        # Compute predictions for all points in predict_X relative to x0
        X_pred_aug = np.vstack([np.ones(len(predict_X)), predict_X - x0]).T
        y_pred = X_pred_aug @ beta
        predictions.append(y_pred)
    
    # Average predictions if multiple models contribute to the same predict_X
    predictions = np.mean(predictions, axis=0)
    
    return predictions


# Step 1: Set up evaluation points and matrices for predictions
def setup_matrices(l, a, a_vals):
    """
    Setup matrix for prediction by combining covariates and treatments.
    """
    n = len(a)
    la_new = np.vstack([np.hstack([l, a[:, None]]),
                        np.hstack([np.repeat(l, len(a_vals), axis=0),
                                   np.tile(a_vals, (n, 1)).flatten()[:, None]])])
    l_new = la_new[:, :-1]  # Remove the treatment variable from la_new for covariates
    return l_new, la_new

# Step 2: Fit SuperLearner to predict treatment and residuals
def fit_models(l, a, y, a_vals):
    """
    Fit SuperLearner to predict treatment values, squared residuals, and outcome.
    """
    l_new, la_new = setup_matrices(l, a, a_vals)
    
    # Local linear fit for predicting a using covariates l
    pimod_vals = local_linear_fit(l, a, l_new[:, :l.shape[1]])  # Only use covariates for predicting a
    
    # Predict squared residuals using covariates l
    sq_res = (a - pimod_vals[:len(a)]) ** 2
    pi2mod_vals = local_linear_fit(l, sq_res, l_new[:, :l.shape[1]])  # Only use covariates for squared residuals
    
    # Predict y using both covariates l and treatment a
    mumod_vals = local_linear_fit(np.hstack([l, a[:, None]]), y, la_new)  # Use covariates + treatment for predicting y
    
    return pimod_vals, pi2mod_vals, mumod_vals, l_new, la_new

# Step 3: Approximation function using spline smoothing
def approx_fn(x, y, z):
    spline = UnivariateSpline(x, y, s=0)
    return spline(z)

# Step 4: Construct pi_hat, varpi_hat, mu_hat, and m_hat
def construct_estimates(pimod_vals, pi2mod_vals, mumod_vals, a_vals, l_new, la_new):
    """
    Construct pi_hat, varpi_hat, mu_hat, and m_hat based on the fitted models.
    """
    n = len(pimod_vals) // 2
    a_std = (la_new[:, -1] - pimod_vals) / np.sqrt(pi2mod_vals)
    
    # Approximate pi_hat using density smoothing
    density, bins = np.histogram(a_std[:n], bins=30, density=True)
    midpoints = (bins[:-1] + bins[1:]) / 2
    pihat_vals = approx_fn(midpoints, density, a_std)
    pihat = pihat_vals[:n]
    
    # Matrix for pi_hat, varpi_hat, mu_hat, m_hat
    pihat_mat = pihat_vals[n:].reshape(n, len(a_vals))
    varpihat = approx_fn(a_vals, np.mean(pihat_mat, axis=0), a_vals)
    varpihat_mat = np.tile(varpihat, (n, 1))
    
    muhat = mumod_vals[:n]
    muhat_mat = mumod_vals[n:].reshape(n, len(a_vals))
    mhat = approx_fn(a_vals, np.mean(muhat_mat, axis=0), a_vals)
    mhat_mat = np.tile(mhat, (n, 1))
    
    return pihat, varpihat, muhat, mhat, varpihat_mat, muhat_mat, mhat_mat

# Step 5: Construct pseudo-outcome
def construct_pseudo_outcome(y, muhat, pihat, varpihat, mhat):
    """
    Construct the pseudo-outcome xi.
    """
    return (y - muhat) / (pihat / varpihat) + mhat

# Step 6: Leave-one-out cross-validation to select bandwidth
def select_bandwidth(a, a_vals, pseudo_out):
    """
    Select bandwidth using leave-one-out cross-validation.
    """
    def cts_eff(out, bw):
        return KernelReg(out, a, var_type='c', bw=[bw]).fit(a_vals)[0]
    
    def w_fn(bw):
        w_avals = []
        for a_val in a_vals:
            a_std = (a - a_val) / bw
            kern_std = norm.pdf(a_std) / bw
            term1 = np.mean(a_std ** 2 * kern_std) * (norm.pdf(0) / bw)
            term2 = np.mean(kern_std) * np.mean(a_std ** 2 * kern_std) - np.mean(a_std * kern_std) ** 2
            w_avals.append(term1 / term2)
        return np.array(w_avals) / len(a)

    def hatvals(bw):
        w_vals = w_fn(bw)
        return approx_fn(a_vals, w_vals, a)

    def loss_fn(h):
        hats = hatvals(h)
        eff = cts_eff(pseudo_out, bw=h)
        return np.mean(((pseudo_out - eff) / (1 - hats)) ** 2)

    result = minimize(loss_fn, x0=1, bounds=[(0.01, 50)], tol=0.01)
    return result.x[0]

# Step 7: Estimate effect curve with optimal bandwidth
def estimate_effect_curve(a, pseudo_out, a_vals, h_opt):
    """
    Estimate effect curve with optimal bandwidth.
    """
    return KernelReg(pseudo_out, a, var_type='c', bw=[h_opt]).fit(a_vals)[0]

# Step 8: Estimate confidence intervals
def estimate_confidence_intervals(a, a_vals, pseudo_out, muhat_mat, mhat_mat, varpihat_mat, h_opt):
    """
    Estimate sandwich-style pointwise confidence intervals.
    """
    se = []
    for a_val in a_vals:
        a_std = (a - a_val) / h_opt
        kern_std = norm.pdf(a_std) / h_opt ** 2
        beta = np.polyfit(a_std, pseudo_out, 1, w=kern_std)
        
        # Dh matrix
        Dh = np.array([[np.mean(kern_std), np.mean(kern_std * a_std)],
                       [np.mean(kern_std * a_std), np.mean(kern_std * a_std ** 2)]])
        
        # Integrals
        kern_mat = np.tile(norm.pdf((a_vals - a_val) / h_opt) / h_opt, (len(a), 1))
        g2 = np.tile((a_vals - a_val) / h_opt, (len(a), 1))
        
        intfn1_mat = kern_mat * (muhat_mat - mhat_mat) * varpihat_mat
        intfn2_mat = g2 * kern_mat * (muhat_mat - mhat_mat) * varpihat_mat
        
        int1 = np.sum(intfn1_mat, axis=1)
        int2 = np.sum(intfn2_mat, axis=1)
        
        # Sandwich variance
        sigma = np.cov(np.linalg.solve(Dh, np.vstack([int1, a_std * int2])))
        se.append(np.sqrt(sigma[0, 0]))
    
    est = estimate_effect_curve(a, pseudo_out, a_vals, h_opt)
    ci_lower = est - 1.96 * np.array(se) / np.sqrt(len(a))
    ci_upper = est + 1.96 * np.array(se) / np.sqrt(len(a))
    
    return est, ci_lower, ci_upper

# Full pipeline to estimate treatment effect and confidence intervals
def estimate_treatment_effect(l, a, y, a_vals):
    """
    Full pipeline to estimate treatment effect and confidence intervals.
    """
    
    # Step 2: Fit models and construct estimates
    pimod_vals, pi2mod_vals, mumod_vals, l_new, la_new = fit_models(l, a, y, a_vals)
    pihat, varpihat, muhat, mhat, varpihat_mat, muhat_mat, mhat_mat = construct_estimates(
        pimod_vals, pi2mod_vals, mumod_vals, a_vals, l_new, la_new)
    
    # Step 5: Construct pseudo-outcomes
    pseudo_out = construct_pseudo_outcome(y, muhat, pihat, varpihat, mhat)
    
    # Step 6: Select bandwidth using cross-validation
    h_opt = select_bandwidth(a, a_vals, pseudo_out)
    
    # Step 7 & 8: Estimate the effect curve and confidence intervals
    est, ci_lower, ci_upper = estimate_confidence_intervals(
        a, a_vals, pseudo_out, muhat_mat, mhat_mat, varpihat_mat, h_opt)
    
    return est, ci_lower, ci_upper, a_vals

