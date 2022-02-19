"""Titanic classification pipeline."""

from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)
from feature_engine.encoding import (
    OneHotEncoder,
    RareLabelEncoder,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from classification_model.config.core import config
from classification_model.processing import ExtractLetterTransformer

# set up the pipeline
titanic_pipe = Pipeline([

    # ===== IMPUTATION =====
    # impute categorical variables with string missing
    ('categorical_imputation', CategoricalImputer(
        imputation_method='missing', 
        variables=config.model_config.categorical_vars
        )
    ),

    # add missing indicator to numerical variables
    ('missing_indicator', AddMissingIndicator(
        variables=config.model_config.numerical_vars
        )
    ),

    # impute numerical variables with the median
    ('median_imputation', MeanMedianImputer(
        imputation_method='median', 
        variables=config.model_config.numerical_vars
        )
    ),


    # Extract letter from cabin
    ('extract_letter', ExtractLetterTransformer(
        variables=config.model_config.cabin_vars
        )
    ),


    # == CATEGORICAL ENCODING ======
    # remove categories present in less than 5% of the observations (0.05)
    # group them in one category called 'Rare'
    ('rare_label_encoder', RareLabelEncoder(
        tol=0.05, 
        n_categories=1, 
        variables=config.model_config.categorical_vars
        )
    ),


    # encode categorical variables using one hot encoding into k-1 variables
    ('categorical_encoder', OneHotEncoder(
        drop_last=True, 
        variables=config.model_config.categorical_vars
        )
    ),

    # scale
    ('scaler', StandardScaler()),

    ('Logit', LogisticRegression(
        C=config.model_config.C, 
        random_state=config.model_config.random_state
        )
    ),
])