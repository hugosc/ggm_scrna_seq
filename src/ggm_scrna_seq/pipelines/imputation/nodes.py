import pandas as pd  # type: ignore
from typing import Iterable, Tuple

from magic import MAGIC
from ggm_scrna_seq.libs.molecular_cross_validation import GridSearchMCV

import logging

log = logging.getLogger(__name__)


def select_strain_and_environment(
    df_counts: pd.DataFrame, strain_group: str, condition_name: str
) -> pd.DataFrame:
    """Subselect condition and strain group for analysis

    Args:
        df_counts (pd.DataFrame): [description]
        strain_group (str): [description]
        condition_name (str): [description]

    Returns:
        pd.DataFrame: [description]
    """
    return df_counts[
        (df_counts["Genotype_Group"] == strain_group)
        & (df_counts["Condition"] == condition_name)
    ]


def select_gene_columns(
    df_counts: pd.DataFrame, columns_to_remove: Iterable
) -> pd.DataFrame:
    log.info(
        f"Removing non-gene count columns {columns_to_remove} from total of {df_counts.shape[1]}"
    )
    dropped = df_counts.drop(columns=columns_to_remove)
    log.info(f"{dropped.shape[1]} columns remain")
    return dropped


def drop_genes_without_expression(df_counts: pd.DataFrame) -> pd.DataFrame:

    total_counts_per_gene = df_counts.sum(axis=0)
    genes_with_some_counts = total_counts_per_gene[total_counts_per_gene > 0].index

    log.info(
        f"Removing {df_counts.shape[1] - len(genes_with_some_counts)} genes with no counts"
    )
    return df_counts[genes_with_some_counts]


def impute_genes_with_cross_validation(
    df_counts: pd.DataFrame, param_grid: dict, capture_efficiency: float
) -> Tuple[pd.DataFrame, dict]:
    mc_validator = GridSearchMCV(
        MAGIC(), param_grid, loss="poisson", sample_ratio=capture_efficiency
    )
    df_imputed = df_counts.copy()
    df_imputed.iloc[:] = mc_validator.fit_transform(df_counts)
    return df_imputed, mc_validator.best_params_
