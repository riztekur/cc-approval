from kedro.pipeline import Pipeline, node, pipeline

from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_dataset,
                inputs=None,
                outputs=["raw_application_record", "raw_credit_record"],
                name="get_dataset_from_kaggle",
            ),
            node(
                func=preprocess_application_record,
                inputs="raw_application_record",
                outputs="preprocessed_application",
                name="preprocess_application_node",
            ),
            node(
                func=preprocess_credit_record,
                inputs="raw_credit_record",
                outputs="preprocessed_credit",
                name="preprocess_credit_node",
            ),
            node(
                func=merge_dataset,
                inputs=["preprocessed_application", "preprocessed_credit"],
                outputs="merged_dataset",
                name="merge_dataset_node",
            ),
            node(
                func=feature_engineering,
                inputs="merged_dataset",
                outputs="engineered_dataset",
                name="feature_engineering_node",
            ),
            node(
                func=feature_selection,
                inputs="engineered_dataset",
                outputs="reduced_dataset",
                name="feature_selection_node",
            ),
            node(
                func=make_bins,
                inputs="reduced_dataset",
                outputs="bins",
                name="make_bins",
            ),
            node(
                func=woe_transformer,
                inputs=["reduced_dataset","bins"],
                outputs="woe_dataset",
                name="create_woe_dataset_node",
            ),
        ]
    )
