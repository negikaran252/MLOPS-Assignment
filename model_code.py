import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn


from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

'''
training code to train the model
'''
def train_model():
    df = pd.read_csv('data/diabetes.csv')
    X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI',
            'DiabetesPedigree','Age']].values, df['Diabetic'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    hyperparam_value=0.001
    regression_model = LogisticRegression(C=hyperparam_value, solver="liblinear").fit(X_train, y_train)

    y_hat = regression_model.predict(X_test)
    acc = np.average(y_hat == y_test)

    y_scores = regression_model.predict_proba(X_test)
    auc = roc_auc_score(y_test,y_scores[:,1])

    print("Accuracy: ",acc)
    return [regression_model,acc,hyperparam_value]

'''
checks if the current model has best metrics than previous models
'''
def is_current_model_best(curr_run_id,curr_accuracy):
    df = mlflow.search_runs()
    exclude_run_ids = [curr_run_id]
    df = df[~df["run_id"].isin(exclude_run_ids)]
    if df.empty:
        return True
    best_model_run_id = df.loc[df['metrics.accuracy'].idxmax()]['run_id']
    best_accuracy=df['metrics.accuracy'].max()
    if curr_accuracy>best_accuracy:
        return True
    return False
    
if __name__ == "__main__":
    experiment_name="MLOPS_ASSIGNMENT"
    run_name=datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_list=train_model()
    reg_model=logs_list[0]
    curr_accuracy=logs_list[1]
    hparam_value=logs_list[2]
    
    # logging the parameters, accuracy, tag and model for each run in the experiment
    curr_run_id=""
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param('param', hparam_value)
        mlflow.log_metric('accuracy', curr_accuracy)
        model_fit=""
        if curr_accuracy>0.9:
            model_fit="overfit"
        elif curr_accuracy>0.75:
            model_fit="good fit"
        elif curr_accuracy>0.5:
            model_fit="low accuracy"
        else:
            model_fit="underfit"
        mlflow.set_tag("Model fit", model_fit)
        mlflow.sklearn.log_model(reg_model, "model")
        curr_run_id=run.info.run_id

    # register the model to model_registry if the current run has best accuracy.
    register_flag=is_current_model_best(curr_run_id,curr_accuracy)
    if register_flag:
        model_name="Diabetes_Predictor"
        with mlflow.start_run(run_id=curr_run_id) as run:
            result=mlflow.register_model(f"runs:/{curr_run_id}/model",model_name)
        print("Done: Registered the current model as best model.New accuracy is: ",curr_accuracy)
        
        # moving the model from staged to production
        client = mlflow.tracking.MlflowClient()
        model_versions = client.get_latest_versions(model_name)
        new_version=0
        for mv in model_versions:
            new_version=mv.version
        print(new_version)
        client.transition_model_version_stage(name=model_name,version=new_version,stage="Production")
    else:
        print("Done: Best model was already registered.")


