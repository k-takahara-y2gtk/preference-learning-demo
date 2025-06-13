import os
import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import datetime
import logging
import time
import joblib
from pathlib import Path
import pandas as pd
import random

# Tensorboard関連のインポート
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        print("Warning: tensorboard not available. Install torch or tensorboardX for tensorboard logging.")
        SummaryWriter = None

def setup_logging():
    """ログ設定を行う関数"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("/workspace/mlr/lgbm/logs", f"training_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Tensorboardログディレクトリ
    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # ログ設定
    log_file = os.path.join(log_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting 100 iterations training at {timestamp}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Tensorboard directory: {tensorboard_dir}")
    
    # TensorBoard writer作成
    tb_writer = None
    if SummaryWriter is not None:
        tb_writer = SummaryWriter(log_dir=tensorboard_dir)
        logger.info("TensorBoard writer initialized")
    else:
        logger.warning("TensorBoard writer not available")
    
    return logger, log_dir, tb_writer

def load_data(fold_num=1):
    """クロスバリデーションのデータ読み込み用関数"""
    root_dir = "/workspace/data/raw/mslr-web"
    fold_dir = os.path.join(root_dir, f"Fold{fold_num}")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from Fold{fold_num}...")
    
    train_file = os.path.join(fold_dir, "train.txt")
    val_file = os.path.join(fold_dir, "vali.txt")
    test_file = os.path.join(fold_dir, "test.txt")
    
    # データ読み込み
    x_train, y_train = load_svmlight_file(train_file)
    x_val, y_val = load_svmlight_file(val_file)
    x_test, y_test = load_svmlight_file(test_file)
    
    def extract_qids_and_create_groups(file_path):
        qids = []
        with open(file_path, "r") as f:
            for line in f:
                parts = line.split()
                for part in parts:
                    if part.startswith("qid:"):
                        qids.append(int(part.split(":")[1]))
                        break
        
        qid_counts = {}
        for qid in qids:
            qid_counts[qid] = qid_counts.get(qid, 0) + 1
        
        query_groups = [qid_counts[qid] for qid in sorted(qid_counts.keys())]
        return query_groups
    
    # クエリグループの作成
    train_query = extract_qids_and_create_groups(train_file)
    val_query = extract_qids_and_create_groups(val_file)
    test_query = extract_qids_and_create_groups(test_file)
    
    logger.info(f"Data loaded: {x_train.shape[0]} train, {x_val.shape[0]} val, {x_test.shape[0]} test examples")
    logger.info(f"Queries: {len(train_query)} train, {len(val_query)} val, {len(test_query)} test")
    
    return (x_train, y_train, train_query), (x_val, y_val, val_query), (x_test, y_test, test_query)

def get_fixed_params():
    """固定のハイパーパラメータを返す関数"""
    # Optunaでやったやつ。
    # Best parameters:
    # num_leaves: 239
    # learning_rate: 0.038054395619039415
    # feature_fraction: 0.6223597094305492
    # bagging_fraction: 0.902323321146222
    # min_child_samples: 62
    # reg_lambda: 1.5006291342706868
    # max_depth: 12
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [3, 5, 10],
        'boosting_type': 'gbdt',
        'num_leaves': 239,
        'learning_rate': 0.038054395619039415,
        'feature_fraction': 0.6223597094305492,
        'bagging_fraction': 0.902323321146222,
        'bagging_freq': 1,
        'min_child_samples': 62,
        'reg_alpha': 0.0,
        'reg_lambda': 1.5006291342706868,
        'max_depth': 12,
        'verbosity': -1,
        'num_threads': -1
    }
    return params

class TensorBoardCallback:
    """LightGBM用のTensorBoardコールバッククラス"""
    
    def __init__(self, tb_writer, iteration, fold_num):
        self.tb_writer = tb_writer
        self.iteration = iteration
        self.fold_num = fold_num
        self.step = 0
        
    def __call__(self, env):
        if self.tb_writer is None:
            return
            
        # 評価結果を取得
        for data_name, eval_name, result, _ in env.evaluation_result_list:
            # タグ名を作成
            tag = f"fold_{self.fold_num}/{data_name}/{eval_name}"
            global_step = self.iteration * 1000 + self.step
            
            # TensorBoardに記録
            self.tb_writer.add_scalar(tag, result, global_step)
            
            # 特別にNDCG@10も記録
            if eval_name == 'ndcg@10':
                self.tb_writer.add_scalar(f"ndcg10_summary/fold_{self.fold_num}", result, global_step)
        
        self.step += 1

def train_single_iteration(iteration, fold_nums, tb_writer, base_seed=42):
    """単一のイテレーションでトレーニングを実行"""
    logger = logging.getLogger(__name__)
    
    # イテレーションごとに異なるシードを設定
    seed = base_seed + iteration
    params = get_fixed_params()
    params['seed'] = seed
    
    results = {
        'iteration': iteration,
        'seed': seed,
        'fold_results': {},
        'avg_ndcg': 0.0
    }
    
    ndcg_scores = []
    
    try:
        for fold_num in fold_nums:
            logger.info(f"Iteration {iteration+1}/100 - Training on fold {fold_num} (seed: {seed})")
            
            train_data, val_data, test_data = load_data(fold_num)
            x_train, y_train, train_query = train_data
            x_val, y_val, val_query = val_data
            x_test, y_test, test_query = test_data
            
            # LightGBMのDataset
            train_set = lgb.Dataset(x_train, label=y_train, group=train_query)
            val_set = lgb.Dataset(x_val, label=y_val, group=val_query, reference=train_set)
            test_set = lgb.Dataset(x_test, label=y_test, group=test_query, reference=train_set)
            
            # Early stopping設定
            early_stopping_callback = lgb.early_stopping(50)
            log_evaluation_callback = lgb.log_evaluation(0)  # ログを非表示
            
            # TensorBoardコールバック
            callbacks = [early_stopping_callback, log_evaluation_callback]
            if tb_writer is not None:
                tb_callback = TensorBoardCallback(tb_writer, iteration, fold_num)
                callbacks.append(tb_callback)
            
            model = lgb.train(
                params,
                train_set,
                valid_sets=[val_set],
                valid_names=["valid"],
                num_boost_round=1000,
                callbacks=callbacks
            )
            
            # 各セットでの評価
            train_pred = model.predict(x_train)
            val_pred = model.predict(x_val)
            test_pred = model.predict(x_test)
            
            # NDCG@10スコアを取得
            val_ndcg = model.best_score['valid']['ndcg@10']
            
            # テストセットでの予測とNDCG計算（参考用）
            test_eval_result = model.predict(x_test)
            
            # 結果を記録
            fold_result = {
                'val_ndcg@10': val_ndcg,
                'best_iteration': model.best_iteration,
                'num_trees': model.num_trees()
            }
            
            results['fold_results'][fold_num] = fold_result
            ndcg_scores.append(val_ndcg)
            
            # TensorBoardに個別結果を記録
            if tb_writer is not None:
                global_step = iteration
                tb_writer.add_scalar(f"iteration_results/fold_{fold_num}_ndcg10", val_ndcg, global_step)
                tb_writer.add_scalar(f"iteration_results/fold_{fold_num}_best_iter", model.best_iteration, global_step)
                tb_writer.add_scalar(f"iteration_results/fold_{fold_num}_num_trees", model.num_trees(), global_step)
            
            logger.info(f"Fold {fold_num} - Val NDCG@10: {val_ndcg:.4f}, Best iteration: {model.best_iteration}")
    
    except Exception as e:
        logger.error(f"Error in iteration {iteration}: {e}")
        results['error'] = str(e)
        return results
    
    # 平均NDCG@10を計算
    results['avg_ndcg'] = np.mean(ndcg_scores)
    results['std_ndcg'] = np.std(ndcg_scores)
    results['min_ndcg'] = np.min(ndcg_scores)
    results['max_ndcg'] = np.max(ndcg_scores)
    
    # TensorBoardに統計値を記録
    if tb_writer is not None:
        global_step = iteration
        tb_writer.add_scalar("iteration_summary/avg_ndcg10", results['avg_ndcg'], global_step)
        tb_writer.add_scalar("iteration_summary/std_ndcg10", results['std_ndcg'], global_step)
        tb_writer.add_scalar("iteration_summary/min_ndcg10", results['min_ndcg'], global_step)
        tb_writer.add_scalar("iteration_summary/max_ndcg10", results['max_ndcg'], global_step)
    
    logger.info(f"Iteration {iteration+1} - Average NDCG@10: {results['avg_ndcg']:.4f} ± {results['std_ndcg']:.4f}")
    
    return results

def save_results(all_results, log_dir):
    """全結果を保存する関数"""
    logger = logging.getLogger(__name__)
    
    # 結果をDataFrameに変換
    results_summary = []
    detailed_results = []
    
    for result in all_results:
        if 'error' in result:
            continue
            
        # サマリー結果
        summary_row = {
            'iteration': result['iteration'],
            'seed': result['seed'],
            'avg_ndcg': result['avg_ndcg'],
            'std_ndcg': result['std_ndcg'],
            'min_ndcg': result['min_ndcg'],
            'max_ndcg': result['max_ndcg']
        }
        results_summary.append(summary_row)
        
        # 詳細結果（フォルドごと）
        for fold_num, fold_result in result['fold_results'].items():
            detail_row = {
                'iteration': result['iteration'],
                'seed': result['seed'],
                'fold': fold_num,
                'val_ndcg@10': fold_result['val_ndcg@10'],
                'best_iteration': fold_result['best_iteration'],
                'num_trees': fold_result['num_trees']
            }
            detailed_results.append(detail_row)
    
    # DataFrameに変換
    summary_df = pd.DataFrame(results_summary)
    detailed_df = pd.DataFrame(detailed_results)
    
    # CSVファイルに保存
    summary_file = os.path.join(log_dir, "results_summary.csv")
    detailed_file = os.path.join(log_dir, "results_detailed.csv")
    
    summary_df.to_csv(summary_file, index=False)
    detailed_df.to_csv(detailed_file, index=False)
    
    logger.info(f"Summary results saved to {summary_file}")
    logger.info(f"Detailed results saved to {detailed_file}")
    
    # 統計情報を保存
    stats_file = os.path.join(log_dir, "statistics.txt")
    with open(stats_file, 'w') as f:
        f.write("100 Iterations Training Statistics\n")
        f.write("="*40 + "\n")
        f.write(f"Total iterations: {len(summary_df)}\n")
        f.write(f"Average NDCG@10: {summary_df['avg_ndcg'].mean():.4f} ± {summary_df['avg_ndcg'].std():.4f}\n")
        f.write(f"Best NDCG@10: {summary_df['avg_ndcg'].max():.4f}\n")
        f.write(f"Worst NDCG@10: {summary_df['avg_ndcg'].min():.4f}\n")
        f.write(f"Median NDCG@10: {summary_df['avg_ndcg'].median():.4f}\n")
        f.write(f"\nFixed parameters:\n")
        params = get_fixed_params()
        for key, value in params.items():
            f.write(f"  {key}: {value}\n")
    
    logger.info(f"Statistics saved to {stats_file}")
    
    return summary_df, detailed_df

def plot_results(summary_df, log_dir):
    """結果をプロットする関数"""
    logger = logging.getLogger(__name__)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('100 Iterations Training Results', fontsize=16)
    
    # 1. イテレーションごとの平均NDCG@10
    axes[0, 0].plot(summary_df['iteration'], summary_df['avg_ndcg'], 'b-', alpha=0.7)
    axes[0, 0].fill_between(summary_df['iteration'], 
                           summary_df['avg_ndcg'] - summary_df['std_ndcg'],
                           summary_df['avg_ndcg'] + summary_df['std_ndcg'], 
                           alpha=0.3)
    axes[0, 0].set_title('Average NDCG@10 per Iteration')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('NDCG@10')
    axes[0, 0].grid(True)
    
    # 2. NDCG@10の分布
    axes[0, 1].hist(summary_df['avg_ndcg'], bins=20, alpha=0.7, color='green')
    axes[0, 1].axvline(summary_df['avg_ndcg'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {summary_df["avg_ndcg"].mean():.4f}')
    axes[0, 1].set_title('Distribution of Average NDCG@10')
    axes[0, 1].set_xlabel('NDCG@10')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. 最大・最小NDCG@10の範囲
    axes[1, 0].fill_between(summary_df['iteration'], 
                           summary_df['min_ndcg'], 
                           summary_df['max_ndcg'], 
                           alpha=0.5, color='orange', label='Min-Max Range')
    axes[1, 0].plot(summary_df['iteration'], summary_df['avg_ndcg'], 'r-', label='Average')
    axes[1, 0].set_title('NDCG@10 Range (Min-Max) per Iteration')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('NDCG@10')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. 標準偏差の推移
    axes[1, 1].plot(summary_df['iteration'], summary_df['std_ndcg'], 'purple', marker='o', markersize=2)
    axes[1, 1].set_title('Standard Deviation of NDCG@10 per Iteration')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Standard Deviation')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_file = os.path.join(log_dir, 'training_results.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Results plot saved to {plot_file}")

def log_final_statistics_to_tensorboard(tb_writer, summary_df, all_results):
    """最終統計をTensorBoardに記録"""
    if tb_writer is None:
        return
    
    successful_results = [r for r in all_results if 'error' not in r]
    avg_scores = [r['avg_ndcg'] for r in successful_results]
    
    # 最終統計をTensorBoardに記録
    tb_writer.add_scalar("final_statistics/overall_mean_ndcg10", np.mean(avg_scores), 0)
    tb_writer.add_scalar("final_statistics/overall_std_ndcg10", np.std(avg_scores), 0)
    tb_writer.add_scalar("final_statistics/overall_max_ndcg10", np.max(avg_scores), 0)
    tb_writer.add_scalar("final_statistics/overall_min_ndcg10", np.min(avg_scores), 0)
    tb_writer.add_scalar("final_statistics/overall_median_ndcg10", np.median(avg_scores), 0)
    tb_writer.add_scalar("final_statistics/successful_iterations", len(successful_results), 0)
    
    # ヒストグラムも追加
    tb_writer.add_histogram("final_statistics/ndcg10_distribution", np.array(avg_scores), 0)

def main():
    """メイン実行関数"""
    # ログ設定
    logger, log_dir, tb_writer = setup_logging()
    
    try:
        # 固定パラメータの表示
        params = get_fixed_params()
        logger.info("Fixed hyperparameters:")
        for key, value in params.items():
            logger.info(f"  {key}: {value}")
        
        # TensorBoardにハイパーパラメータを記録
        if tb_writer is not None:
            # ハイパーパラメータをテキストとして記録
            hparams_text = "\n".join([f"{k}: {v}" for k, v in params.items()])
            tb_writer.add_text("hyperparameters", hparams_text, 0)
        
        logger.info("Starting 100 iterations training...")
        start_time = time.time()
        
        # 使用するフォルド
        fold_nums = [1, 2, 3]  # 3つのフォルドで評価
        logger.info(f"Using folds: {fold_nums}")
        
        # 100回のイテレーション実行
        all_results = []
        for i in range(100):
            result = train_single_iteration(i, fold_nums, tb_writer)
            all_results.append(result)
            
            # 10回ごとに中間結果を表示
            if (i + 1) % 10 == 0:
                completed_results = [r for r in all_results if 'error' not in r]
                if completed_results:
                    avg_scores = [r['avg_ndcg'] for r in completed_results]
                    current_mean = np.mean(avg_scores)
                    current_std = np.std(avg_scores)
                    logger.info(f"Progress: {i+1}/100 - Current average: {current_mean:.4f} ± {current_std:.4f}")
                    
                    # TensorBoardに中間統計を記録
                    if tb_writer is not None:
                        tb_writer.add_scalar("progress/mean_ndcg10", current_mean, i+1)
                        tb_writer.add_scalar("progress/std_ndcg10", current_std, i+1)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
        
        # 結果保存
        summary_df, detailed_df = save_results(all_results, log_dir)
        
        # プロット作成
        plot_results(summary_df, log_dir)
        
        # TensorBoardに最終統計を記録
        log_final_statistics_to_tensorboard(tb_writer, summary_df, all_results)
        
        # 最終統計の表示
        successful_results = [r for r in all_results if 'error' not in r]
        avg_scores = [r['avg_ndcg'] for r in successful_results]
        
        print("\n" + "="*60)
        print("100 ITERATIONS TRAINING COMPLETED")
        print("="*60)
        print(f"Successful iterations: {len(successful_results)}/100")
        print(f"Average NDCG@10: {np.mean(avg_scores):.4f} ± {np.std(avg_scores):.4f}")
        print(f"Best NDCG@10: {np.max(avg_scores):.4f}")
        print(f"Worst NDCG@10: {np.min(avg_scores):.4f}")
        print(f"Median NDCG@10: {np.median(avg_scores):.4f}")
        print(f"Results saved in: {log_dir}")
        
        # パラメータ表示
        print(f"\nFixed hyperparameters used:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        if tb_writer is not None:
            print(f"\nTensorBoard logs saved in: {os.path.join(log_dir, 'tensorboard')}")
            print("To view TensorBoard, run:")
            print(f"tensorboard --logdir={os.path.join(log_dir, 'tensorboard')}")
            tb_writer.close()
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        if tb_writer is not None:
            tb_writer.close()
        raise

if __name__ == "__main__":
    main()