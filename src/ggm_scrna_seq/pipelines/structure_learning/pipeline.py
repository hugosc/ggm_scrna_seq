from kedro.pipeline import Pipeline, node
from .nodes import gaussian_graphical_model_learn, robust_covariance_estimation


def create_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                gaussian_graphical_model_learn,
                ["gene_counts_imputed.csv", "params:ggm_params"],
                ["precision_structure.pkl", "precision_p_values.pkl"],
                tags=["structure_learning"],
            ),
            node(
                robust_covariance_estimation,
                "gene_counts_imputed.csv",
                "empirical_covariance.pkl",
                tags=["structure_learning", "covariance_estimation"]
            )
        ]
    )
