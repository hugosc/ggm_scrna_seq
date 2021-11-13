import pandas as pd  # type: ignore
import numpy as np

from typing import Tuple

from rpy2.robjects import packages as rpacks  # type: ignore
from rpy2.robjects import StrVector, r, numpy2ri  # type: ignore

from sklearn.covariance import LedoitWolf


import logging

log = logging.getLogger(__name__)


def install_rpackage(package_name):
    utils = rpacks.importr("utils")
    utils.chooseCRANmirror(ind=1) # pylint: disable=maybe-no-member

    if not rpacks.isinstalled(package_name):
        utils.install_packages(StrVector([package_name]))


def gaussian_graphical_model_learn(
    df_expression: pd.DataFrame, ggm_params: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """[summary]

    Args:
        df_expression (pd.DataFrame): [description]
        ggm_params (dict): [description]

    Returns:
        Tuple[np.ndarray, np.ndarray]: [description]
    """    
    install_rpackage("SILGGM")
    silggm = rpacks.importr("SILGGM")

    numpy2ri.activate()
    nrow, ncol = df_expression.shape
    r_m_imputed = r.matrix(df_expression.values, nrow=nrow, ncol=ncol)
    results = silggm.SILGGM(r_m_imputed) # pylint: disable=maybe-no-member

    p_values : np.ndarray = results[-1]
    structure = p_values <= 0.05

    numpy2ri.deactivate()

    return structure, p_values

def robust_covariance_estimation(df_expression: pd.DataFrame):

    log.info("Fitting Ledoit Wolf estimator for shrinked covariance")

    cov_estimator = LedoitWolf()
    cov_estimator.fit(df_expression)
    
    log.info(f"performed shrinkage with factor {cov_estimator.shrinkage_}")
    return cov_estimator.covariance_
