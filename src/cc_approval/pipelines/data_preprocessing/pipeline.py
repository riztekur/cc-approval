from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_application_record, preprocess_credit_record, merge_dataset, feature_selection, woe_transformer


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_application_record,
                inputs="application_record",
                outputs="preprocessed_application",
                name="preprocess_application_node",
            ),
            node(
                func=preprocess_credit_record,
                inputs="credit_record",
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
                func=feature_selection,
                inputs="merged_dataset",
                outputs="reduced_dataset",
                name="feature_selection_node",
            ),
            node(
                func=woe_transformer,
                inputs="reduced_dataset",
                outputs="woe_dataset",
                name="create_woe_dataset_node",
            ),
        ]
    )
