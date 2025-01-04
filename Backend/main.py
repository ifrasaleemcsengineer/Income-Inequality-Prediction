import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import warnings
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dataframe = None

def preprocess_dataframe(df):
    """Preprocess the dataframe by dropping columns with too many missing values and encoding categorical features."""
    try:
        nan_cols_drop = [col for col in df.columns if df[col].isna().sum() / df.shape[0] > 0.5]
        df = df.drop(columns=nan_cols_drop, axis=1)
        if 'household_stat' in df.columns:
            df.drop(columns=['household_stat'], axis=1, inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error preprocessing dataframe: {e}")
        raise HTTPException(status_code=400, detail=f"Error preprocessing dataframe: {e}")

@app.get("/get-data-head/")
def get_data_head(row_limit: int = Query(5, ge=1)):
    global dataframe
    try:
        dataframe = pd.read_csv("data.csv")
        dataframe = dataframe.where(pd.notnull(dataframe), None)  
        head = dataframe.head(row_limit).to_dict(orient='records')
        logger.info(f"Successfully retrieved top {row_limit} rows of data.")
        return {"head": head}
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")


@app.get("/get-data-summary/")
def get_data_summary():
    global dataframe
    if dataframe is None:
        logger.warning("No data loaded when attempting to retrieve summary.")
        raise HTTPException(status_code=400, detail="No data uploaded")

    try:
        def clean_value(val):
            if isinstance(val, (float, np.floating)):
                if np.isnan(val) or np.isinf(val):
                    return None
            return val

        total_rows = dataframe.shape[0]
        total_columns = dataframe.shape[1]
        numerical_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = dataframe.select_dtypes(include=["object", "category"]).columns.tolist()

        total_missing = dataframe.isnull().sum().sum()
        percentage_missing = (total_missing / (total_rows * total_columns)) * 100
        rows_with_missing = dataframe.isnull().any(axis=1).sum()

        numerical_data = dataframe[numerical_columns]
        overall_mean = clean_value(numerical_data.mean().mean())
        overall_std = clean_value(numerical_data.std().mean())
        overall_min = clean_value(numerical_data.min().min())
        overall_max = clean_value(numerical_data.max().max())

        categorical_data = dataframe[categorical_columns]

        excluded_columns = [
            "migration_code_change_in_msa",
            "migration_code_move_within_reg",
            "migration_code_change_in_reg",
            "migration_prev_sunbelt"
        ]
        filtered_categorical_columns = [
            col for col in categorical_columns if col not in excluded_columns
        ]

        most_frequent_value = dataframe[filtered_categorical_columns].mode().iloc[0].to_dict()
        total_unique_values = categorical_data.nunique().sum()

        summary = {
            "general_info": {
                "total_rows": total_rows,
                "total_columns": total_columns,
                "numerical_columns": len(numerical_columns),
                "categorical_columns": len(categorical_columns),
            },
            "data_completeness": {
                "total_missing_values": int(total_missing),
                "percentage_missing": round(percentage_missing, 2),
                "rows_with_missing_values": int(rows_with_missing),
            },
            "numerical_data": {
                "overall_mean": round(overall_mean, 2) if overall_mean else None,
                "overall_std": round(overall_std, 2) if overall_std else None,
                "overall_min": round(overall_min, 2) if overall_min else None,
                "overall_max": round(overall_max, 2) if overall_max else None,
            },
            "categorical_data": {
                "most_frequent_value": most_frequent_value,
                "total_unique_values": int(total_unique_values),
            },
        }

        return summary

    except Exception as e:
        logger.error(f"Error summarizing data: {e}")
        raise HTTPException(status_code=400, detail=f"Error summarizing data: {e}")



@app.get("/predict/{model_type}/")
def predict_f1_score(model_type: str):
    print(model_type)
    global dataframe
    if dataframe is None:
        logger.warning("No data loaded when attempting to train model.")
        raise HTTPException(status_code=400, detail="Data not loaded. Please load data first by calling /get-data-head.")

    try:
        df = preprocess_dataframe(dataframe)
        X, y = df.drop(['ID', 'income_above_limit'], axis=1, errors='ignore'), df['income_above_limit']

        target_mapping = {'Above limit': 1, 'Below limit': 0}
        y = y.map(target_mapping)

        if y.isnull().any():
            logger.error("Target mapping resulted in null values.")
            raise HTTPException(status_code=400, detail="Invalid target values in the dataset.")

        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        columns_to_encode = [
            'gender', 'education', 'marital_status', 'race', 'is_hispanic',
            'employment_commitment', 'industry_code_main', 'household_summary',
            'tax_status', 'citizenship', 'country_of_birth_own', 'country_of_birth_father',
            'country_of_birth_mother', 'migration_code_change_in_msa',
            'migration_prev_sunbelt', 'migration_code_move_within_reg', 'migration_code_change_in_reg'
        ]
        columns_to_keep = X_train.drop(columns_to_encode, axis=1).columns

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_train_encoded = pd.concat([
            pd.DataFrame(encoder.fit_transform(X_train[columns_to_encode]),
                         columns=encoder.get_feature_names_out(columns_to_encode)),
            X_train[columns_to_keep].reset_index(drop=True)
        ], axis=1)

        X_test_encoded = pd.concat([
            pd.DataFrame(encoder.transform(X_test[columns_to_encode]),
                         columns=encoder.get_feature_names_out(columns_to_encode)),
            X_test[columns_to_keep].reset_index(drop=True)
        ], axis=1)

        # Select model
        if model_type.lower() == "random_forest":
            model = RandomForestClassifier(random_state=42)
        elif model_type.lower() == "xgb":
            model = XGBClassifier()
        elif model_type.lower() == "gradient_boosting":
            model = GradientBoostingClassifier(random_state=42)
        else:
            logger.error(f"Invalid model type: {model_type}")
            raise HTTPException(status_code=400, detail="Invalid model type")

        # Train and predict
        model.fit(X_train_encoded, y_train)
        y_pred = model.predict(X_test_encoded)
        f1 = round(f1_score(y_test, y_pred), 2)

        logger.info(f"Model {model_type} achieved F1 score: {f1}")
        return {"model_type": model_type, "f1_score": f1}
    except Exception as e:
        logger.error(f"Error training {model_type} model: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing the data: {e}")


@app.get("/get-chart-data/")
def get_chart_data():
    try:
        models = ['Random Forest', 'XGBClassifier', 'Gradient Boosting']
        f1_scores = [0.99, 0.91, 0.87]
        data = {"models": models, "f1_scores": f1_scores}
        logger.info("Successfully retrieved chart data.")
        return data
    except Exception as e:
        logger.error(f"Error generating chart data: {e}")
        raise HTTPException(status_code=400, detail=f"Error generating chart data: {e}")

@app.get("/distributions")
def get_distributions():
    global dataframe
    try:
        total_employed = dataframe['total_employed'].value_counts().to_dict()
        
        income = dataframe['income_above_limit'].value_counts().to_dict()
        
        citizenship = dataframe['citizenship'].value_counts().to_dict()
        
        education = dataframe['education'].value_counts().to_dict()
        
        response = {
            "total_employed": total_employed,
            "income_above_limit": income,
            "citizenship": citizenship,
            "education": education
        }
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
