import joblib
import sklearn
import emlearn

model = joblib.load("best_XGBoost.joblib")
rf = model.named_steps['classifier']

cmodel = emlearn.convert(rf, method='inline')

cmodel.save(
    file="xgboost_model.h",
    name="xgboost_model"
)