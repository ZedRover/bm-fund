import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def objective(trial):
    # 加载数据
    df_train = pd.read_feather("../data/Xtrain.feather")
    y_train = pd.read_feather("../data/Ytrain.feather").values.ravel()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        df_train, y_train, test_size=0.2, shuffle=False
    )
    clip = trial.suggest_int("clip", 0, 10)
    y_train = np.clip(y_train, -clip, clip)
    # 定义参数空间
    param = {
        "objective": "regression",
        "metric": "l2",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 20, 80),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.9),
        "bagging_freq": trial.suggest_int("bagging_freq", 5, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "num_threads": 60,
    }

    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # 训练模型
    gbm = lgb.train(
        param,
        train_data,
        valid_sets=[valid_data],
        verbose_eval=False,
        early_stopping_rounds=6000,
    )

    # 预测
    preds = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    # 计算IC和R2
    ic = spearmanr(y_test, preds).correlation
    r2 = r2_score(y_test, preds)

    # 优化目标是最大化IC和R2的平均值
    return ic, r2


study = optuna.create_study(
    directions=["maximize", "maximize"], storage="sqlite:///lgb_study.db"
)
# study = optuna.load_study(study_name="lgb_study", storage="sqlite:///lgb_study.db")
study.optimize(objective, n_trials=100)

print("Number of finished trials:", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Values: IC={}, R2={}".format(trial.values[0], trial.values[1]))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
