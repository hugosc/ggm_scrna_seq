def select_strain_and_environment(df_counts, strain_group, condition_name):
    return df_counts[
        (df_counts["Genotype_Group"] == strain_group) & 
        (df_counts["Condition"] == condition_name)
    ]