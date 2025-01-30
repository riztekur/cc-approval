from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["woe_dataset", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "params:xgboost_classifier"],
                outputs="regressor",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["regressor", "bins", "X_train", "X_test", "y_train", "y_test"],
                outputs=["performance_summary","train_performance_viz","test_performance_viz"],
                name="evaluate_model_node",
            ),
        ]
    )