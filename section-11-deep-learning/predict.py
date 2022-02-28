import data_management as dm


def make_prediction(*, path_to_images) -> float:
    """Make a prediction using the saved model pipeline."""

    dataframe = path_to_images
    pipe = dm.load_pipeline_keras()
    predictions = pipe.pipe.predict(dataframe)

    return predictions