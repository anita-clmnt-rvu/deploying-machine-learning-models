from sklearn.pipeline import Pipeline

import config
import preprocessors as pp
import model

pipe = Pipeline([
    ('dataset', pp.CreateDataset(config.IMAGE_SIZE)),
    ('cnn_model', model.cnn_clf)
])

if __name__=='__main__':

    from sklearn.metrics import accuracy_score
    import data_management as dm
    import config

    images_df = dm.load_image_paths(config.DATA_FOLDER)
    X_train, X_test, y_train, y_test = dm.get_train_test_target(images_df)

    enc = pp.TargetEncoder()
    enc.fit(y_train)
    y_train = enc.transform(y_train)

    pipe.pipe.fit(X_train, y_train)