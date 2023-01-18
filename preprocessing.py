import numpy as np


def preprocess_ou(residuals, r2_threshold=None, tol=1e-6):
    """Fits Ornstein-Uhlenbeck process to a dataframe of residuals."""
    # Train AR(1) model on cumulative residuals
    cumulative_residuals = np.cumsum(residuals, axis=0)
    Xs = cumulative_residuals[:-1]
    Ys = cumulative_residuals[1:]

    meansX, meansY = np.mean(Xs, axis=0), np.mean(Ys, axis=0)
    varsX, varsY = np.var(Xs, axis=0), np.var(Ys, axis=0)
    covXY = np.mean((Xs - meansX.reshape(1, -1)) * (Ys - meansY.reshape(1, -1)), axis=0)

    # Estimate the a and b parameters for the AR(1) model, and compute R-squared
    b_vec = covXY / varsX
    a_vec = meansY - b_vec * meansX
    r2 = covXY**2 / (varsX * varsY)

    # Use the above estimates to compute s-scores
    ar_residuals = Ys - b_vec.reshape(1, -1) * Xs - a_vec.reshape(1, -1)
    mu_vec = a_vec / (1 - b_vec + tol)
    sigma_kappa_vec = np.sqrt(np.var(ar_residuals, axis=0) / (1 - b_vec**2 + tol))
    s_scores = (cumulative_residuals[-1] - mu_vec) / sigma_kappa_vec
    if r2_threshold:
        s_scores[r2 < r2_threshold] = np.nan

    return s_scores
