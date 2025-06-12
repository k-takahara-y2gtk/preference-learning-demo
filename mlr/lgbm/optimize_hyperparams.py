import os
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import datetime
import logging
import time
import joblib
from pathlib import Path
import pandas as pd

def setup_logging():
    """ログ設定を行う関数"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("/workspace/mlr/lgbm/logs", f"optuna_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # ログ設定
    log_file = os.path.join(log_dir, "optimization.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting hyperparameter optimization at {timestamp}")
    logger.info(f"Log directory: {log_dir}")
    
    return logger, log_dir

def load_data(fold_num=1):
    """クロスバリデーションのデータ読み込み用関数"""
    root_dir = "/workspace/data/raw/mslr-web"
    fold_dir = os.path.join(root_dir, f"Fold{fold_num}")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from Fold{fold_num}...")
    
    train_file = os.path.join(fold_dir, "train.txt")
    val_file = os.path.join(fold_dir, "vali.txt")
    
    # データ読み込み
    x_train, y_train = load_svmlight_file(train_file)
    x_val, y_val = load_svmlight_file(val_file)
    
    # クエリIDの抽出
    train_qids = []
    with open(train_file, "r") as f:
        for line in f:
            parts = line.split()
            for part in parts:
                if part.startswith("qid:"):
                    train_qids.append(int(part.split(":")[1]))
                    break
    
    val_qids = []
    with open(val_file, "r") as f:
        for line in f:
            parts = line.split()
            for part in parts:
                if part.startswith("qid:"):
                    val_qids.append(int(part.split(":")[1]))
                    break
    
    # グループ情報の作成
    train_qid_counts = {}
    for qid in train_qids:
        if qid not in train_qid_counts:
            train_qid_counts[qid] = 0
        train_qid_counts[qid] += 1
    
    val_qid_counts = {}
    for qid in val_qids:
        if qid not in val_qid_counts:
            val_qid_counts[qid] = 0
        val_qid_counts[qid] += 1
    
    train_query = []
    for qid in sorted(train_qid_counts.keys()):
        train_query.append(train_qid_counts[qid])
    
    val_query = []
    for qid in sorted(val_qid_counts.keys()):
        val_query.append(val_qid_counts[qid])
    
    logger.info(f"Data loaded: {x_train.shape[0]} training examples, {x_val.shape[0]} validation examples")
    logger.info(f"Number of training queries: {len(train_query)}")
    logger.info(f"Number of validation queries: {len(val_query)}")
    
    return x_train, y_train, x_val, y_val, train_query, val_query

def objective(trial, fold_nums=[1, 2]):
    """
    最適化の目的関数
    """
    logger = logging.getLogger(__name__)
    
    # ハイパーパラメータの提案
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [3, 5, 10],
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': 1,
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': 0.0,
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'verbosity': -1,
        'seed': 42
    }
    
    ndcg_scores = []
    
    try:
        for fold_num in fold_nums:
            logger.info(f"Training on fold {fold_num}")
            x_train, y_train, x_val, y_val, train_query, val_query = load_data(fold_num)
            
            # LightGBMのDataset
            train_set = lgb.Dataset(x_train, label=y_train, group=train_query)
            val_set = lgb.Dataset(x_val, label=y_val, group=val_query, reference=train_set)
            
            # Early stopping設定
            early_stopping_callback = lgb.early_stopping(50)
            log_evaluation_callback = lgb.log_evaluation(0)  # ログを非表示
            
            model = lgb.train(
                params,
                train_set,
                valid_sets=[val_set],
                valid_names=["valid"],
                num_boost_round=1000,
                callbacks=[early_stopping_callback, log_evaluation_callback]
            )
            
            # 最終的なNDCG@10スコアを取得
            best_score = model.best_score['valid']['ndcg@10']
            ndcg_scores.append(best_score)
            
            logger.info(f"Fold {fold_num} NDCG@10: {best_score:.4f}")
    
    except Exception as e:
        logger.error(f"Error in trial: {e}")
        return 0.0
    
    # 平均NDCG@10を返す（最大化したいので負の値は返さない）
    avg_ndcg = np.mean(ndcg_scores)
    logger.info(f"Average NDCG@10: {avg_ndcg:.4f}")
    
    return avg_ndcg

def save_results(study, log_dir):
    """最適化結果を保存する関数"""
    logger = logging.getLogger(__name__)
    
    # 結果をDataFrameに変換
    trials_df = study.trials_dataframe()
    
    # CSVファイルに保存
    results_file = os.path.join(log_dir, "optimization_results.csv")
    trials_df.to_csv(results_file, index=False)
    logger.info(f"Results saved to {results_file}")
    
    # 最適なパラメータを保存
    best_params_file = os.path.join(log_dir, "best_params.txt")
    with open(best_params_file, 'w') as f:
        f.write(f"Best value: {study.best_value:.4f}\n")
        f.write("Best params:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
    
    # studyオブジェクトを保存
    study_file = os.path.join(log_dir, "study.pkl")
    joblib.dump(study, study_file)
    logger.info(f"Study object saved to {study_file}")
    
    return trials_df

def plot_optimization_history(study, log_dir):
    """最適化履歴をプロットする関数"""
    try:
        import optuna.visualization as vis
        
        # 最適化履歴
        fig1 = vis.plot_optimization_history(study)
        fig1.write_html(os.path.join(log_dir, "optimization_history.html"))
        
        # パラメータの重要度
        fig2 = vis.plot_param_importances(study)
        fig2.write_html(os.path.join(log_dir, "param_importances.html"))
        
        # パラレル座標プロット
        fig3 = vis.plot_parallel_coordinate(study)
        fig3.write_html(os.path.join(log_dir, "parallel_coordinate.html"))
        
        logging.getLogger(__name__).info("Plots saved to HTML files")
        
    except ImportError:
        # matplotlibでのシンプルなプロット
        plt.figure(figsize=(10, 6))
        trials = study.trials
        values = [trial.value for trial in trials if trial.value is not None]
        plt.plot(values)
        plt.title('Optimization History')
        plt.xlabel('Trial')
        plt.ylabel('NDCG@10')
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, 'optimization_history.png'))
        plt.close()
        
        logging.getLogger(__name__).info("Basic plot saved to PNG file")

def main():
    """メイン実行関数"""
    # ログ設定
    logger, log_dir = setup_logging()
    
    try:
        # Optuna study作成
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42), # ベイズ最適化的にパラメータ探索する
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30)# 性能が悪い時に探索を打ち切る
        ) 
        
        logger.info("Starting hyperparameter optimization...")
        start_time = time.time()
        
        # 最適化実行
        study.optimize(
            lambda trial: objective(trial, fold_nums=[1, 2, 3, 4, 5]),  # 使用するフォルドを指定
            n_trials=130,  # 試行回数
            show_progress_bar=True
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Optimization completed in {elapsed_time:.2f} seconds")
        
        # 結果表示
        logger.info(f"Best value: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        # 結果保存
        trials_df = save_results(study, log_dir)
        
        # プロット作成
        plot_optimization_history(study, log_dir)
        
        # 統計情報
        logger.info(f"Number of finished trials: {len(study.trials)}")
        logger.info(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        
        print("\n" + "="*50)
        print("OPTIMIZATION COMPLETED")
        print("="*50)
        print(f"Best NDCG@10: {study.best_value:.4f}")
        print(f"Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(f"Results saved in: {log_dir}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()