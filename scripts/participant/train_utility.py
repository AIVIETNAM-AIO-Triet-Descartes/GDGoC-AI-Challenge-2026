import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

def train_lgb_model(df, target_col, objective, metric, output_json_path):
    # Action-specific features only (excl. state-wide features like step, power_score, bombs_left, etc.)
    features = [
        "norm_item", "norm_box", "norm_kill", "norm_pressure", "norm_survive", "norm_mobility", "norm_territory", "norm_danger",
        "raw_item", "raw_box", "raw_kill", "raw_pressure", "raw_survive", "raw_mobility", "raw_territory", "raw_danger",
        "action", "boxes_hit", "items_hit"
    ]
    
    # Check that all features exist
    for f in features:
        if f not in df.columns:
            raise ValueError(f"Feature '{f}' not found in the dataset.")
            
    # Group Shuffle Split based on match_id to avoid data leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['match_id']))
    
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    
    X_train = df_train[features]
    y_train = df_train[target_col]
    X_val = df_val[features]
    y_val = df_val[target_col]
    
    train_dataset = lgb.Dataset(X_train, label=y_train)
    val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)
    
    params = {
        'objective': objective,
        'metric': metric,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.8,
        'verbose': -1,
        'random_state': 42
    }
    
    print(f"Training LightGBM model for target '{target_col}'...")
    
    # Train the model
    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=300,
        valid_sets=[train_dataset, val_dataset],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=True),
            lgb.log_evaluation(period=50)
        ]
    )
    
    # Compute feature importance
    importance = model.feature_importance(importance_type='gain')
    feat_imp = pd.Series(importance, index=features).sort_values(ascending=False)
    print("\nFeature Importances (Gain):")
    print(feat_imp.head(10))
    print("-" * 50)
    
    # Dump model to JSON format
    model_json = model.dump_model()
    
    # Write to target path
    out_file = Path(output_json_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(model_json, f, indent=2)
        
    print(f"Model successfully saved to {out_file}\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train learned utility models using LightGBM.")
    parser.add_argument("--dataset_path", type=str, default="agent/learned_utility_dataset.csv", help="Path to CSV dataset.")
    parser.add_argument("--rank_model_path", type=str, default="agent/rank_model.json", help="Output path for rank prediction model.")
    parser.add_argument("--win_model_path", type=str, default="agent/win_model.json", help="Output path for win prediction model.")
    parser.add_argument("--survival_model_path", type=str, default="agent/survival_model.json", help="Output path for discounted survival model.")
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return
        
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded. Shape: {df.shape}")
    
    # Calculate discounted survival target
    print("Calculating discounted survival target...")
    df['total_steps'] = df.groupby(['match_id', 'agent_id'])['step'].transform('max')
    gamma = 0.96
    steps_to_death = df['total_steps'] - df['step']
    df['discounted_survival'] = np.where(
        df['rank'] == 0,
        1.0,
        1.0 - (gamma ** steps_to_death)
    )
    
    # Train Rank Regressor (MSE loss)
    train_lgb_model(
        df=df,
        target_col="rank",
        objective="regression",
        metric="rmse",
        output_json_path=args.rank_model_path
    )
    
    # Train Win Regressor (Binary Logloss)
    train_lgb_model(
        df=df,
        target_col="win",
        objective="binary",
        metric="binary_logloss",
        output_json_path=args.win_model_path
    )
    
    # Train Discounted Survival Regressor (MSE loss)
    train_lgb_model(
        df=df,
        target_col="discounted_survival",
        objective="regression",
        metric="rmse",
        output_json_path=args.survival_model_path
    )

if __name__ == "__main__":
    main()
