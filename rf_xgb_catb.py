import os
import time
import numpy as np
import pandas as pd
import matplotlib

# 使用非交互后端，避免 PyCharm/backend_interagg 报错
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor
from catboost import CatBoostRegressor


# =========================================================
# 0. 创建输出文件夹
# =========================================================
output_dir = "model_outputs"
os.makedirs(output_dir, exist_ok=True)


# =========================================================
# 1. 开始计时
# =========================================================
start = time.time()
print("start_time:", start)


# =========================================================
# 2. 读取数据
# =========================================================
data = pd.read_excel("data_set.xlsx")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
feature_names = list(X.columns)

print(f"Dataset shape: X={X.shape}, y={y.shape}")


# =========================================================
# 3. 外层随机划分设置
# =========================================================
random_seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# =========================================================
# 4. 定义模型及参数空间
# =========================================================
model_specs = {
    "RF": {
        "model": RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        ),
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"]
        }
    },

    "XGB": {
        "model": XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42,
            n_jobs=1
        ),
        "param_grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }
    },

    "CatB": {
        "model": CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            verbose=0,
            allow_writing_files=False,
            thread_count=1
        ),
        "param_grid": {
            "iterations": [100, 200, 300],
            "depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "l2_leaf_reg": [1, 3, 5]
        }
    }
}


# =========================================================
# 5. 总汇总表
# =========================================================
all_model_summary = []


# =========================================================
# 6. 主循环：依次运行 RF / XGB / CatB
# =========================================================
for model_name, spec in model_specs.items():

    print("\n" + "=" * 80)
    print(f"Starting model: {model_name}")
    print("=" * 80)

    base_model = spec["model"]
    param_grid = spec["param_grid"]

    results = []
    best_params_list = []
    feature_importance_list = []

    # -----------------------------------------------------
    # 外层 10 次随机划分
    # -----------------------------------------------------
    for split_id, seed in enumerate(random_seeds, start=1):
        print(f"\n[{model_name}] Outer Split {split_id}/{len(random_seeds)} | random_state={seed}")

        # 外层划分：先隔离测试集
        X_train_outer, X_test_outer, y_train_outer, y_test_outer = train_test_split(
            X, y,
            test_size=0.25,
            random_state=seed
        )

        # 内层CV：仅在训练集上调参
        inner_cv = KFold(
            n_splits=5,
            shuffle=True,
            random_state=seed
        )

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=inner_cv,
            n_jobs=-1,
            refit=True
        )

        # 只在外层训练集上搜索最优参数
        grid_search.fit(X_train_outer, y_train_outer)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print("Best params:", best_params)

        # 外层测试集预测
        y_pred = best_model.predict(X_test_outer)

        # 指标
        mae = mean_absolute_error(y_test_outer, y_pred)
        mse = mean_squared_error(y_test_outer, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_outer, y_pred)

        results.append({
            "split_id": split_id,
            "random_seed": seed,
            "model": model_name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        })

        best_params_list.append({
            "split_id": split_id,
            "random_seed": seed,
            **best_params
        })

        # 特征重要性
        if hasattr(best_model, "feature_importances_"):
            feature_importance_list.append(best_model.feature_importances_)
        else:
            # 理论上这三个模型都有 feature_importances_
            feature_importance_list.append(np.zeros(X.shape[1]))

        print(
            f"{model_name} --> "
            f"MAE: {mae:.4f}, "
            f"MSE: {mse:.4f}, "
            f"RMSE: {rmse:.4f}, "
            f"R2: {r2:.4f}"
        )

    # =====================================================
    # 7. 保存每个模型每次 split 结果
    # =====================================================
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, f"{model_name}_split_results.csv")
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

    # 最优参数
    best_params_df = pd.DataFrame(best_params_list)
    best_params_path = os.path.join(output_dir, f"{model_name}_best_params_each_split.csv")
    best_params_df.to_csv(best_params_path, index=False, encoding="utf-8-sig")

    # =====================================================
    # 8. 计算 mean ± SD
    # =====================================================
    summary_row = {
        "model": model_name,

        "MAE_mean": results_df["MAE"].mean(),
        "MAE_std": results_df["MAE"].std(),

        "MSE_mean": results_df["MSE"].mean(),
        "MSE_std": results_df["MSE"].std(),

        "RMSE_mean": results_df["RMSE"].mean(),
        "RMSE_std": results_df["RMSE"].std(),

        "R2_mean": results_df["R2"].mean(),
        "R2_std": results_df["R2"].std()
    }

    summary_df = pd.DataFrame([summary_row])
    summary_path = os.path.join(output_dir, f"{model_name}_summary_mean_std.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    all_model_summary.append(summary_row)

    print(f"\n========== {model_name} Summary: Mean ± SD ==========")
    print(f"MAE  = {summary_row['MAE_mean']:.4f} ± {summary_row['MAE_std']:.4f}")
    print(f"MSE  = {summary_row['MSE_mean']:.4f} ± {summary_row['MSE_std']:.4f}")
    print(f"RMSE = {summary_row['RMSE_mean']:.4f} ± {summary_row['RMSE_std']:.4f}")
    print(f"R2   = {summary_row['R2_mean']:.4f} ± {summary_row['R2_std']:.4f}")

    # =====================================================
    # 9. 平均特征重要性
    # =====================================================
    feature_importance_array = np.array(feature_importance_list)
    mean_feature_importance = feature_importance_array.mean(axis=0)
    std_feature_importance = feature_importance_array.std(axis=0)

    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance_Mean": mean_feature_importance,
        "Importance_SD": std_feature_importance
    }).sort_values(by="Importance_Mean", ascending=True)

    fi_csv_path = os.path.join(output_dir, f"{model_name}_feature_importance_mean_std.csv")
    feature_importance_df.to_csv(fi_csv_path, index=False, encoding="utf-8-sig")

    # 画图并保存
    plt.figure(figsize=(10, 6))
    plt.barh(
        feature_importance_df["Feature"],
        feature_importance_df["Importance_Mean"],
        xerr=feature_importance_df["Importance_SD"],
        color="skyblue",
        edgecolor="black",
        capsize=3
    )
    plt.xlabel("Mean Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"{model_name} Mean Feature Importances over 10 Outer Splits")
    plt.tight_layout()

    fi_fig_path = os.path.join(output_dir, f"{model_name}_feature_importance_mean_std.png")
    plt.savefig(fi_fig_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# 10. 保存所有模型总汇总表
# =========================================================
all_model_summary_df = pd.DataFrame(all_model_summary)
all_summary_path = os.path.join(output_dir, "all_models_summary_mean_std.csv")
all_model_summary_df.to_csv(all_summary_path, index=False, encoding="utf-8-sig")

print("\n" + "=" * 80)
print("All models summary")
print("=" * 80)
print(all_model_summary_df)


# =========================================================
# 11. 结束计时
# =========================================================
end = time.time()
print("\nend_time:", end)
print("run_time:", end - start)