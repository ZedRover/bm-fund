import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

# 加载数据
df_train = pd.read_feather("../data/Xtrain.feather")
y_train = pd.read_feather("../data/Ytrain.feather").values.ravel()

# 定义测试集和训练集的切分比例
n_samples = len(df_train)
n_train = int(n_samples * 0.8)

# 训练集和测试集的划分
X_train_val = df_train.iloc[:n_train]
y_train_val = y_train[:n_train]
X_test = df_train.iloc[n_train:]
y_test = y_train[n_train:]

# KFold 设置
kf = KFold(n_splits=4, shuffle=True, random_state=42)

# 设置模型参数
params = {
    "objective": "regression",
    "metric": "l2",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
}

# 训练模型并进行交叉验证
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
    X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
    y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    print(f"Training on fold {fold + 1}")
    gbm = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=1000,
        # early_stopping_rounds=50,
        # verbose_eval=100,
    )

    y_pred_val = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    ic = spearmanr(y_val, y_pred_val).correlation
    r2 = r2_score(y_val, y_pred_val)

    print(f"Fold {fold + 1}: IC = {ic:.4f}, R2 = {r2:.4f}")

# 使用全部训练数据重新训练模型
final_model = lgb.train(
    params,
    lgb.Dataset(X_train_val, label=y_train_val),
    num_boost_round=1000,
    verbose_eval=100,
)

# 保存最终模型
final_model.save_model("final_model.txt")

# 测试集预测
y_test_pred = final_model.predict(X_test, num_iteration=final_model.best_iteration)
test_ic = spearmanr(y_test, y_test_pred).correlation
test_r2 = r2_score(y_test, y_test_pred)

print(f"Test Set: IC = {test_ic:.4f}, R2 = {test_r2:.4f}")
