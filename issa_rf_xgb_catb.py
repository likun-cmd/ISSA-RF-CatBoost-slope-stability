import os
import time
import numpy as np
import pandas as pd
import matplotlib

# 非交互后端，避免 backend_interagg 报错
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor


# =========================================================
# 0. 输出目录
# =========================================================
output_dir = "ISSA_nested_outputs"
os.makedirs(output_dir, exist_ok=True)


# =========================================================
# 1. 数据读取
# =========================================================
def load_data(path):
    if path.endswith(".xlsx") or path.endswith(".xls"):
        data = pd.read_excel(path)
    else:
        data = pd.read_csv(path)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y


# =========================================================
# 2. Tent 混沌映射改进的 SSA
# =========================================================
class ChaoticSSA:
    def __init__(self, objective_func, dim, lb, ub, pop_size=20, max_iter=20):
        self.objective_func = objective_func
        self.dim = dim
        self.lb = np.array(lb, dtype=float)
        self.ub = np.array(ub, dtype=float)
        self.pop_size = pop_size
        self.max_iter = max_iter

        self.population = None
        self.fitness = None
        self.best_fitness = float("inf")
        self.best_solution = None
        self.convergence = []

    def tent_map(self, x, mu=1.99):
        return mu * min(x, 1 - x)

    def chaotic_sequence(self, length, seed=0.7):
        seq = np.zeros(length)
        seq[0] = seed
        for i in range(1, length):
            seq[i] = self.tent_map(seq[i - 1])
        return seq

    def initialize(self):
        chaos = self.chaotic_sequence(self.pop_size * self.dim).reshape(self.pop_size, self.dim)
        self.population = self.lb + (self.ub - self.lb) * chaos
        self.fitness = np.array([self.objective_func(ind) for ind in self.population])

        best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]

    def update(self, iteration):
        PD = max(1, int(0.2 * self.pop_size))
        SD = max(1, int(0.1 * self.pop_size))

        # 发现者
        for i in range(PD):
            chaos = self.chaotic_sequence(self.dim, seed=np.random.rand())
            for j in range(self.dim):
                if np.random.rand() < 0.8:
                    alpha = max(np.random.rand(), 1e-8)
                    self.population[i, j] *= np.exp(-iteration / (alpha * self.max_iter))
                else:
                    self.population[i, j] += chaos[j] * (self.best_solution[j] - self.population[i, j])

            self.population[i] = np.clip(self.population[i], self.lb, self.ub)
            new_fit = self.objective_func(self.population[i])

            if new_fit < self.fitness[i]:
                self.fitness[i] = new_fit
                if new_fit < self.best_fitness:
                    self.best_fitness = new_fit
                    self.best_solution = self.population[i].copy()

        # 跟随者
        for i in range(PD, self.pop_size):
            A = np.random.randint(0, 2, size=self.dim) * 2 - 1

            if i > self.pop_size / 2:
                self.population[i] += A * (self.best_solution - self.population[i]) / (i ** 2)
            else:
                self.population[i] = self.best_solution + 0.5 * A * np.abs(self.population[i] - self.best_solution)

            self.population[i] = np.clip(self.population[i], self.lb, self.ub)
            new_fit = self.objective_func(self.population[i])

            if new_fit < self.fitness[i]:
                self.fitness[i] = new_fit
                if new_fit < self.best_fitness:
                    self.best_fitness = new_fit
                    self.best_solution = self.population[i].copy()

        # 警戒者
        for _ in range(SD):
            idx = np.random.randint(self.pop_size)
            chaos = self.chaotic_sequence(self.dim, seed=np.random.rand())
            self.population[idx] += chaos * (self.best_solution - self.population[idx])
            self.population[idx] = np.clip(self.population[idx], self.lb, self.ub)

            new_fit = self.objective_func(self.population[idx])

            if new_fit < self.fitness[idx]:
                self.fitness[idx] = new_fit
                if new_fit < self.best_fitness:
                    self.best_fitness = new_fit
                    self.best_solution = self.population[idx].copy()

        self.convergence.append(self.best_fitness)

    def optimize(self, verbose=True):
        self.initialize()
        if verbose:
            print(f"  Initial best RMSE: {self.best_fitness:.6f}")

        for iteration in range(self.max_iter):
            self.update(iteration)
            if verbose:
                print(f"  Iter {iteration + 1}/{self.max_iter} | Best RMSE: {self.best_fitness:.6f}")

        return self.best_solution, self.convergence


# =========================================================
# 3. 参数解码函数
# =========================================================
def decode_rf_params(params):
    max_features_raw = params[4]
    if max_features_raw <= 1.0:
        max_features = max(min(float(max_features_raw), 1.0), 0.1)
    else:
        max_features = int(np.round(max_features_raw))

    return {
        "n_estimators": int(np.round(params[0])),
        "max_depth": int(np.round(params[1])),
        "min_samples_leaf": int(np.round(params[2])),
        "min_samples_split": int(np.round(params[3])),
        "max_features": max_features,
        "random_state": 42,
        "n_jobs": -1
    }


def decode_xgb_params(params):
    return {
        "max_depth": int(np.round(params[0])),
        "learning_rate": float(params[1]),
        "n_estimators": int(np.round(params[2])),
        "gamma": float(params[3]),
        "subsample": float(params[4]),
        "colsample_bytree": float(params[5]),
        "min_child_weight": float(params[6]),
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
        "tree_method": "hist"
    }


def decode_catb_params(params):
    return {
        "iterations": int(np.round(params[0])),
        "learning_rate": max(float(params[1]), 0.01),
        "depth": int(np.round(params[2])),
        "l2_leaf_reg": max(float(params[3]), 0.0),
        "border_count": int(np.round(params[4])),
        "random_strength": max(float(params[5]), 1e-9),
        "grow_policy": "SymmetricTree" if params[6] < 0.5 else "Depthwise",
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "verbose": 0,
        "random_seed": 42,
        "allow_writing_files": False
    }


# =========================================================
# 4. 构造 ISSA 目标函数：仅在外层训练集上做 5-fold CV
# =========================================================
def make_objective(model_name, X_train_outer, y_train_outer, inner_cv):
    def objective(params):
        try:
            if model_name == "ISSA_RF":
                model = RandomForestRegressor(**decode_rf_params(params))
            elif model_name == "ISSA_XGB":
                model = XGBRegressor(**decode_xgb_params(params))
            elif model_name == "ISSA_CatB":
                model = CatBoostRegressor(**decode_catb_params(params))
            else:
                raise ValueError("Unknown model name")

            scores = cross_val_score(
                model,
                X_train_outer,
                y_train_outer,
                scoring="neg_mean_squared_error",
                cv=inner_cv,
                n_jobs=-1
            )

            return np.sqrt(-np.mean(scores))

        except Exception:
            return float("inf")

    return objective


# =========================================================
# 5. 运行单个模型的 nested repeated random splits
# =========================================================
def run_nested_issa_model(
    model_name,
    X,
    y,
    feature_names,
    random_seeds,
    dim,
    lb,
    ub,
    pop_size,
    max_iter
):
    results = []
    best_params_list = []
    feature_importance_list = []

    for split_id, seed in enumerate(random_seeds, start=1):
        print("\n" + "-" * 80)
        print(f"{model_name} | Outer Split {split_id}/{len(random_seeds)} | random_state={seed}")
        print("-" * 80)

        # 外层：隔离测试集
        X_train_outer, X_test_outer, y_train_outer, y_test_outer = train_test_split(
            X, y,
            test_size=0.25,
            random_state=seed
        )

        # 内层CV：只在外层训练集上优化参数
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=seed)

        objective_func = make_objective(model_name, X_train_outer, y_train_outer, inner_cv)

        optimizer = ChaoticSSA(
            objective_func=objective_func,
            dim=dim,
            lb=lb,
            ub=ub,
            pop_size=pop_size,
            max_iter=max_iter
        )

        best_solution, convergence = optimizer.optimize(verbose=True)

        # 解码最优参数
        if model_name == "ISSA_RF":
            best_params = decode_rf_params(best_solution)
            best_model = RandomForestRegressor(**best_params)

        elif model_name == "ISSA_XGB":
            best_params = decode_xgb_params(best_solution)
            best_model = XGBRegressor(**best_params)

        elif model_name == "ISSA_CatB":
            best_params = decode_catb_params(best_solution)
            best_model = CatBoostRegressor(**best_params)

        else:
            raise ValueError("Unknown model name")

        print("Best params:", best_params)
        print(f"Best inner CV RMSE: {optimizer.best_fitness:.6f}")

        # 用最优参数在整个外层训练集重新训练
        best_model.fit(X_train_outer, y_train_outer)

        # 在外层测试集评估
        y_pred = best_model.predict(X_test_outer)

        mae = mean_absolute_error(y_test_outer, y_pred)
        mse = mean_squared_error(y_test_outer, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_outer, y_pred)

        results.append({
            "split_id": split_id,
            "random_seed": seed,
            "model": model_name,
            "inner_best_RMSE": optimizer.best_fitness,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        })

        bp = {
            "split_id": split_id,
            "random_seed": seed,
            **best_params
        }
        best_params_list.append(bp)

        if hasattr(best_model, "feature_importances_"):
            feature_importance_list.append(best_model.feature_importances_)
        else:
            try:
                feature_importance_list.append(best_model.get_feature_importance())
            except Exception:
                feature_importance_list.append(np.zeros(len(feature_names)))

        print(
            f"{model_name} --> "
            f"MAE: {mae:.4f}, "
            f"MSE: {mse:.4f}, "
            f"RMSE: {rmse:.4f}, "
            f"R2: {r2:.4f}"
        )

        # 每个 split 的收敛曲线
        plt.figure(figsize=(8, 5))
        plt.plot(convergence, color="red", linewidth=2)
        plt.xlabel("Iteration")
        plt.ylabel("Best RMSE")
        plt.title(f"{model_name} Convergence | Split {split_id}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{model_name}_split_{split_id}_convergence.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    # -----------------------------------------------------
    # 保存每次 split 结果
    # -----------------------------------------------------
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(output_dir, f"{model_name}_split_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    best_params_df = pd.DataFrame(best_params_list)
    best_params_df.to_csv(
        os.path.join(output_dir, f"{model_name}_best_params_each_split.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # -----------------------------------------------------
    # 计算 mean ± SD
    # -----------------------------------------------------
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
    summary_df.to_csv(
        os.path.join(output_dir, f"{model_name}_summary_mean_std.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n========== {model_name} Summary: Mean ± SD ==========")
    print(f"MAE  = {summary_row['MAE_mean']:.4f} ± {summary_row['MAE_std']:.4f}")
    print(f"MSE  = {summary_row['MSE_mean']:.4f} ± {summary_row['MSE_std']:.4f}")
    print(f"RMSE = {summary_row['RMSE_mean']:.4f} ± {summary_row['RMSE_std']:.4f}")
    print(f"R2   = {summary_row['R2_mean']:.4f} ± {summary_row['R2_std']:.4f}")

    # -----------------------------------------------------
    # 平均特征重要性
    # -----------------------------------------------------
    fi_array = np.array(feature_importance_list)
    fi_mean = fi_array.mean(axis=0)
    fi_std = fi_array.std(axis=0)

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance_Mean": fi_mean,
        "Importance_SD": fi_std
    }).sort_values(by="Importance_Mean", ascending=True)

    fi_df.to_csv(
        os.path.join(output_dir, f"{model_name}_feature_importance_mean_std.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    plt.figure(figsize=(10, 6))
    plt.barh(
        fi_df["Feature"],
        fi_df["Importance_Mean"],
        xerr=fi_df["Importance_SD"],
        color="skyblue",
        edgecolor="black",
        capsize=3
    )
    plt.xlabel("Mean Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"{model_name} Mean Feature Importance over 10 Outer Splits")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{model_name}_feature_importance_mean_std.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    return summary_row


# =========================================================
# 6. 主程序
# =========================================================
if __name__ == "__main__":
    start_time = time.time()
    print("start_time:", start_time)

    # 数据
    data_path = "data_set.xlsx"
    X, y = load_data(data_path)
    feature_names = list(X.columns)

    print(f"Data loaded: X={X.shape}, y={y.shape}")

    # 外层随机种子
    random_seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    all_model_summary = []

    # =====================================================
    # ISSA-RF
    # =====================================================
    rf_summary = run_nested_issa_model(
        model_name="ISSA_RF",
        X=X,
        y=y,
        feature_names=feature_names,
        random_seeds=random_seeds,
        dim=5,
        lb=[10, 1, 1, 2, 0.1],      # n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features
        ub=[200, 50, 20, 20, 10],
        pop_size=20,
        max_iter=20
    )
    all_model_summary.append(rf_summary)

    # =====================================================
    # ISSA-XGB
    # =====================================================
    xgb_summary = run_nested_issa_model(
        model_name="ISSA_XGB",
        X=X,
        y=y,
        feature_names=feature_names,
        random_seeds=random_seeds,
        dim=7,
        lb=[3, 0.01, 50, 0.0, 0.5, 0.5, 1.0],   # max_depth, learning_rate, n_estimators, gamma, subsample, colsample_bytree, min_child_weight
        ub=[10, 0.3, 200, 0.5, 1.0, 1.0, 10.0],
        pop_size=20,
        max_iter=20
    )
    all_model_summary.append(xgb_summary)

    # =====================================================
    # ISSA-CatBoost
    # =====================================================
    catb_summary = run_nested_issa_model(
        model_name="ISSA_CatB",
        X=X,
        y=y,
        feature_names=feature_names,
        random_seeds=random_seeds,
        dim=7,
        lb=[500, 0.01, 4, 1, 32, 0.1, 0],      # iterations, learning_rate, depth, l2_leaf_reg, border_count, random_strength, grow_policy
        ub=[2000, 0.3, 15, 10, 255, 10, 1],
        pop_size=15,
        max_iter=15
    )
    all_model_summary.append(catb_summary)

    # =====================================================
    # 全部模型汇总
    # =====================================================
    all_summary_df = pd.DataFrame(all_model_summary)
    all_summary_df.to_csv(
        os.path.join(output_dir, "ISSA_all_models_summary_mean_std.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 80)
    print("All ISSA models summary")
    print("=" * 80)
    print(all_summary_df)

    end_time = time.time()
    print("\nend_time:", end_time)
    print("run_time:", end_time - start_time)