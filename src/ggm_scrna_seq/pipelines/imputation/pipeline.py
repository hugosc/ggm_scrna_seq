from kedro.pipeline import Pipeline, node

from .nodes import select_strain_and_environment


def create_pipeline() -> Pipeline:
    """Creates imputation pipeline
        1. subselection of environment and strain
        2. molecular cross-validation
        3. imputed data
    Returns:
        Pipeline: Imputation pipeline
    """    
    return Pipeline([
        node(
            select_strain_and_environment,
            [
                "gene_counts.csv", 
                "params:genotype_group",
                "params:condition"
            ],
            "selected_gene_counts.csv",
            tags=["imputation"]
        )
    ])