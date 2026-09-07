import os
import time
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor
from catboost import CatBoostRegressor


# =========================================================
# 0. Output directory
# =========================================================
output_dir = "ISSA_stacking_nested_outputs"
os.makedirs(output_dir, exist_ok=True)


# =========================================================
# 1. Data loading
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
# 2. Chaotic SSA
# =========================================================
class ChaoticSSA:
    def __init__(self, objective_func, dim, lb, ub, pop_size=15, max_iter=15):
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

    def optimize(self, verbose=False, prefix=""):
        self.initialize()
        if verbose:
            print(f"{prefix}Initial best RMSE: {self.best_fitness:.6f}")

        for iteration in range(self.max_iter):
            self.update(iteration)
            if verbose:
                print(f"{prefix}Iter {iteration + 1}/{self.max_iter} | Best RMSE: {self.best_fitness:.6f}")

        return self.best_solution, self.convergence


# =========================================================
# 3. Parameter decoding
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
# 4. Objective function for single-model ISSA
# =========================================================
def make_single_objective(model_type, X_train_outer, y_train_outer, inner_cv):
    def objective(params):
        try:
            if model_type == "RF":
                model = RandomForestRegressor(**decode_rf_params(params))
            elif model_type == "XGB":
                model = XGBRegressor(**decode_xgb_params(params))
            elif model_type == "CatB":
                model = CatBoostRegressor(**decode_catb_params(params))
            else:
                raise ValueError("Unknown model_type")

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
# 5. Optimize a single base learner
# =========================================================
def optimize_base_model(model_type, X_train_outer, y_train_outer, inner_cv, verbose=False):
    if model_type == "RF":
        dim = 5
        lb = [10, 1, 1, 2, 0.1]
        ub = [200, 50, 20, 20, 10]
        pop_size = 15
        max_iter = 15

    elif model_type == "XGB":
        dim = 7
        lb = [3, 0.01, 50, 0.0, 0.5, 0.5, 1.0]
        ub = [10, 0.3, 200, 0.5, 1.0, 1.0, 10.0]
        pop_size = 15
        max_iter = 15

    elif model_type == "CatB":
        dim = 7
        lb = [500, 0.01, 4, 1, 32, 0.1, 0]
        ub = [2000, 0.3, 15, 10, 255, 10, 1]
        pop_size = 12
        max_iter = 12

    else:
        raise ValueError("Unknown model_type")

    optimizer = ChaoticSSA(
        objective_func=make_single_objective(model_type, X_train_outer, y_train_outer, inner_cv),
        dim=dim,
        lb=lb,
        ub=ub,
        pop_size=pop_size,
        max_iter=max_iter
    )

    best_solution, convergence = optimizer.optimize(verbose=verbose, prefix=f"[{model_type}] ")

    if model_type == "RF":
        best_params = decode_rf_params(best_solution)
        model = RandomForestRegressor(**best_params)

    elif model_type == "XGB":
        best_params = decode_xgb_params(best_solution)
        model = XGBRegressor(**best_params)

    elif model_type == "CatB":
        best_params = decode_catb_params(best_solution)
        model = CatBoostRegressor(**best_params)

    return model, best_params, optimizer.best_fitness, convergence


# =========================================================
# 6. Run stacking model with nested outer splits
# =========================================================
def run_nested_stacking_model(model_name, base_model_types, X, y, feature_names, random_seeds):
    results = []
    best_params_list = []
    feature_importance_list = []

    for split_id, seed in enumerate(random_seeds, start=1):
        print("\n" + "=" * 90)
        print(f"{model_name} | Outer Split {split_id}/{len(random_seeds)} | random_state={seed}")
        print("=" * 90)

        # Outer train-test split
        X_train_outer, X_test_outer, y_train_outer, y_test_outer = train_test_split(
            X, y,
            test_size=0.25,
            random_state=seed
        )

        inner_cv = KFold(n_splits=5, shuffle=True, random_state=seed)

        estimators = []
        split_param_record = {
            "split_id": split_id,
            "random_seed": seed
        }

        split_inner_rmse = {}

        # Optimize each base learner sequentially
        for base_type in base_model_types:
            print(f"\nOptimizing base learner: {base_type}")
            base_model, base_params, base_best_rmse, base_convergence = optimize_base_model(
                base_type,
                X_train_outer,
                y_train_outer,
                inner_cv,
                verbose=False
            )

            estimators.append((base_type.lower(), base_model))
            split_inner_rmse[f"{base_type}_inner_best_RMSE"] = base_best_rmse

            for k, v in base_params.items():
                split_param_record[f"{base_type}_{k}"] = v

            # Save convergence curve for each base learner in each split
            plt.figure(figsize=(8, 5))
            plt.plot(base_convergence, linewidth=2)
            plt.xlabel("Iteration")
            plt.ylabel("Best RMSE")
            plt.title(f"{model_name} | Split {split_id} | {base_type} convergence")
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"{model_name}_split_{split_id}_{base_type}_convergence.png"),
                dpi=300,
                bbox_inches="tight"
            )
            plt.close()

            print(f"{base_type} best inner CV RMSE: {base_best_rmse:.6f}")
            print(f"{base_type} best params: {base_params}")

        # Stacking meta-learner
        stacking_model = StackingRegressor(
            estimators=estimators,
            final_estimator=LinearRegression(),
            cv=5,
            n_jobs=-1,
            passthrough=False
        )

        # Train stacking model
        stacking_model.fit(X_train_outer, y_train_outer)

        # Evaluate on test set
        y_pred = stacking_model.predict(X_test_outer)

        mae = mean_absolute_error(y_test_outer, y_pred)
        mse = mean_squared_error(y_test_outer, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_outer, y_pred)

        result_row = {
            "split_id": split_id,
            "random_seed": seed,
            "model": model_name,
            **split_inner_rmse,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }
        results.append(result_row)

        best_params_list.append(split_param_record)

        print(
            f"{model_name} --> "
            f"MAE: {mae:.4f}, "
            f"MSE: {mse:.4f}, "
            f"RMSE: {rmse:.4f}, "
            f"R2: {r2:.4f}"
        )

        # Approximate feature importance: average over base models that support importance extraction
        temp_importances = []
        for _, est in stacking_model.named_estimators_.items():
            if hasattr(est, "feature_importances_"):
                temp_importances.append(est.feature_importances_)
            else:
                try:
                    temp_importances.append(est.get_feature_importance())
                except Exception:
                    pass

        if len(temp_importances) > 0:
            feature_importance_list.append(np.mean(temp_importances, axis=0))
        else:
            feature_importance_list.append(np.zeros(len(feature_names)))

    # Save split results
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

    # Summary statistics
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

    # Average feature importance
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
# 7. Main program
# =========================================================
if __name__ == "__main__":
    start_time = time.time()
    print("start_time:", start_time)

    data_path = "data_set.xlsx"
    X, y = load_data(data_path)
    feature_names = list(X.columns)

    print(f"Data loaded: X={X.shape}, y={y.shape}")

    random_seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    all_summary = []

    # 1. ISSA_RF-XGB
    summary_1 = run_nested_stacking_model(
        model_name="ISSA_RF-XGB",
        base_model_types=["RF", "XGB"],
        X=X,
        y=y,
        feature_names=feature_names,
        random_seeds=random_seeds
    )
    all_summary.append(summary_1)

    # 2. ISSA_RF-CatB
    summary_2 = run_nested_stacking_model(
        model_name="ISSA_RF-CatB",
        base_model_types=["RF", "CatB"],
        X=X,
        y=y,
        feature_names=feature_names,
        random_seeds=random_seeds
    )
    all_summary.append(summary_2)

    # 3. ISSA_XGB-CatB
    summary_3 = run_nested_stacking_model(
        model_name="ISSA_XGB-CatB",
        base_model_types=["XGB", "CatB"],
        X=X,
        y=y,
        feature_names=feature_names,
        random_seeds=random_seeds
    )
    all_summary.append(summary_3)

    # 4. ISSA_RF-XGB-CatB
    summary_4 = run_nested_stacking_model(
        model_name="ISSA_RF-XGB-CatB",
        base_model_types=["RF", "XGB", "CatB"],
        X=X,
        y=y,
        feature_names=feature_names,
        random_seeds=random_seeds
    )
    all_summary.append(summary_4)

    # Save overall summary
    all_summary_df = pd.DataFrame(all_summary)
    all_summary_df.to_csv(
        os.path.join(output_dir, "ISSA_stacking_all_models_summary_mean_std.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 90)
    print("All stacking ISSA models summary")
    print("=" * 90)
    print(all_summary_df)

    end_time = time.time()
    print("\nend_time:", end_time)
    print("run_time:", end_time - start_time)
