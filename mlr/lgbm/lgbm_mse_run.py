import os
import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import ndcg_score
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
    
    logger.info(f"Starting 100 iterations training with MSE loss and NDCG evaluation at {timestamp}")
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
    params = {
        'objective': 'regression',  # MSE損失関数で0-4の関連度を予測
        'metric': ['rmse', 'mae'],  # 学習時の評価メトリクス
        'boosting_type': 'gbdt',
        'num_leaves': 64,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_child_samples': 20,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'max_depth': 7,
        'verbosity': -1,
        'num_threads': -1
    }
    return params

def calculate_ndcg_scores(y_true, y_pred, query_groups, k_values=[3, 5, 10]):
    """
    クエリグループごとにNDCGを計算し、平均を返す
    """
    ndcg_results = {f'ndcg@{k}': [] for k in k_values}
    
    start_idx = 0
    for group_size in query_groups:
        end_idx = start_idx + group_size
        
        # 現在のクエリの真の関連度と予測値
        query_true = y_true[start_idx:end_idx]
        query_pred = y_pred[start_idx:end_idx]
        
        # 各kについてNDCGを計算
        for k in k_values:
            if len(query_true) >= k:  # クエリサイズがk以上の場合のみ計算
                # ndcg_scoreは2次元配列を期待するため、reshapeする
                true_relevance = query_true.reshape(1, -1)
                pred_scores = query_pred.reshape(1, -1)
                
                ndcg_k = ndcg_score(true_relevance, pred_scores, k=k)
                ndcg_results[f'ndcg@{k}'].append(ndcg_k)
        
        start_idx = end_idx
    
    # 平均NDCGを計算
    avg_ndcg = {}
    for k in k_values:
        if ndcg_results[f'ndcg@{k}']:
            avg_ndcg[f'ndcg@{k}'] = np.mean(ndcg_results[f'ndcg@{k}'])
        else:
            avg_ndcg[f'ndcg@{k}'] = 0.0
    
    return avg_ndcg

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
            
            # 特別にRMSEも記録
            if eval_name == 'rmse':
                self.tb_writer.add_scalar(f"rmse_summary/fold_{self.fold_num}", result, global_step)
        
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
        'avg_ndcg@3': 0.0,
        'avg_ndcg@5': 0.0,
        'avg_ndcg@10': 0.0,
        'avg_rmse': 0.0
    }
    
    ndcg3_scores = []
    ndcg5_scores = []
    ndcg10_scores = []
    rmse_scores = []
    
    try:
        for fold_num in fold_nums:
            logger.info(f"Iteration {iteration+1}/100 - Training on fold {fold_num} (seed: {seed})")
            
            train_data, val_data, test_data = load_data(fold_num)
            x_train, y_train, train_query = train_data
            x_val, y_val, val_query = val_data
            x_test, y_test, test_query = test_data
            
            # LightGBMのDataset（MSEではgroupは不要）
            train_set = lgb.Dataset(x_train, label=y_train)
            val_set = lgb.Dataset(x_val, label=y_val, reference=train_set)
            test_set = lgb.Dataset(x_test, label=y_test, reference=train_set)
            
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
            
            # 各セットでの予測
            train_pred = model.predict(x_train)
            val_pred = model.predict(x_val)
            test_pred = model.predict(x_test)
            
            # RMSEスコアを取得
            val_rmse = model.best_score['valid']['rmse']
            
            # 検証セットでNDCGを計算
            val_ndcg_scores = calculate_ndcg_scores(y_val, val_pred, val_query, k_values=[3, 5, 10])
            
            # テストセットでもNDCGを計算（参考用）
            test_ndcg_scores = calculate_ndcg_scores(y_test, test_pred, test_query, k_values=[3, 5, 10])
            
            # 結果を記録
            fold_result = {
                'val_rmse': val_rmse,
                'val_ndcg@3': val_ndcg_scores['ndcg@3'],
                'val_ndcg@5': val_ndcg_scores['ndcg@5'],
                'val_ndcg@10': val_ndcg_scores['ndcg@10'],
                'test_ndcg@3': test_ndcg_scores['ndcg@3'],
                'test_ndcg@5': test_ndcg_scores['ndcg@5'],
                'test_ndcg@10': test_ndcg_scores['ndcg@10'],
                'best_iteration': model.best_iteration,
                'num_trees': model.num_trees()
            }
            
            results['fold_results'][fold_num] = fold_result
            ndcg3_scores.append(val_ndcg_scores['ndcg@3'])
            ndcg5_scores.append(val_ndcg_scores['ndcg@5'])
            ndcg10_scores.append(val_ndcg_scores['ndcg@10'])
            rmse_scores.append(val_rmse)
            
            # TensorBoardに個別結果を記録
            if tb_writer is not None:
                global_step = iteration
                tb_writer.add_scalar(f"iteration_results/fold_{fold_num}_rmse", val_rmse, global_step)
                tb_writer.add_scalar(f"iteration_results/fold_{fold_num}_ndcg@3", val_ndcg_scores['ndcg@3'], global_step)
                tb_writer.add_scalar(f"iteration_results/fold_{fold_num}_ndcg@5", val_ndcg_scores['ndcg@5'], global_step)
                tb_writer.add_scalar(f"iteration_results/fold_{fold_num}_ndcg@10", val_ndcg_scores['ndcg@10'], global_step)
                tb_writer.add_scalar(f"iteration_results/fold_{fold_num}_best_iter", model.best_iteration, global_step)
                tb_writer.add_scalar(f"iteration_results/fold_{fold_num}_num_trees", model.num_trees(), global_step)
                
                # テストセットのNDCGも記録
                tb_writer.add_scalar(f"test_results/fold_{fold_num}_ndcg@3", test_ndcg_scores['ndcg@3'], global_step)
                tb_writer.add_scalar(f"test_results/fold_{fold_num}_ndcg@5", test_ndcg_scores['ndcg@5'], global_step)
                tb_writer.add_scalar(f"test_results/fold_{fold_num}_ndcg@10", test_ndcg_scores['ndcg@10'], global_step)
            
            logger.info(f"Fold {fold_num} - RMSE: {val_rmse:.4f}, NDCG@10: {val_ndcg_scores['ndcg@10']:.4f}, Best iter: {model.best_iteration}")
    
    except Exception as e:
        logger.error(f"Error in iteration {iteration}: {e}")
        results['error'] = str(e)
        return results
    
    # 平均値を計算
    results['avg_rmse'] = np.mean(rmse_scores)
    results['avg_ndcg@3'] = np.mean(ndcg3_scores)
    results['avg_ndcg@5'] = np.mean(ndcg5_scores)
    results['avg_ndcg@10'] = np.mean(ndcg10_scores)
    
    results['std_rmse'] = np.std(rmse_scores)
    results['std_ndcg@3'] = np.std(ndcg3_scores)
    results['std_ndcg@5'] = np.std(ndcg5_scores)
    results['std_ndcg@10'] = np.std(ndcg10_scores)
    
    # TensorBoardに統計値を記録
    if tb_writer is not None:
        global_step = iteration
        tb_writer.add_scalar("iteration_summary/avg_rmse", results['avg_rmse'], global_step)
        tb_writer.add_scalar("iteration_summary/avg_ndcg@3", results['avg_ndcg@3'], global_step)
        tb_writer.add_scalar("iteration_summary/avg_ndcg@5", results['avg_ndcg@5'], global_step)
        tb_writer.add_scalar("iteration_summary/avg_ndcg@10", results['avg_ndcg@10'], global_step)
        tb_writer.add_scalar("iteration_summary/std_rmse", results['std_rmse'], global_step)
        tb_writer.add_scalar("iteration_summary/std_ndcg@10", results['std_ndcg@10'], global_step)
    
    logger.info(f"Iteration {iteration+1} - RMSE: {results['avg_rmse']:.4f}, NDCG@10: {results['avg_ndcg@10']:.4f} ± {results['std_ndcg@10']:.4f}")
    
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
            'avg_rmse': result['avg_rmse'],
            'avg_ndcg@3': result['avg_ndcg@3'],
            'avg_ndcg@5': result['avg_ndcg@5'],
            'avg_ndcg@10': result['avg_ndcg@10'],
            'std_rmse': result['std_rmse'],
            'std_ndcg@3': result['std_ndcg@3'],
            'std_ndcg@5': result['std_ndcg@5'],
            'std_ndcg@10': result['std_ndcg@10']
        }
        results_summary.append(summary_row)
        
        # 詳細結果（フォルドごと）
        for fold_num, fold_result in result['fold_results'].items():
            detail_row = {
                'iteration': result['iteration'],
                'seed': result['seed'],
                'fold': fold_num,
                'val_rmse': fold_result['val_rmse'],
                'val_ndcg@3': fold_result['val_ndcg@3'],
                'val_ndcg@5': fold_result['val_ndcg@5'],
                'val_ndcg@10': fold_result['val_ndcg@10'],
                'test_ndcg@3': fold_result['test_ndcg@3'],
                'test_ndcg@5': fold_result['test_ndcg@5'],
                'test_ndcg@10': fold_result['test_ndcg@10'],
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
        f.write("100 Iterations Training Statistics (MSE Loss with NDCG Evaluation)\n")
        f.write("="*60 + "\n")
        f.write(f"Total iterations: {len(summary_df)}\n")
        f.write(f"Average RMSE: {summary_df['avg_rmse'].mean():.4f} ± {summary_df['avg_rmse'].std():.4f}\n")
        f.write(f"Best RMSE: {summary_df['avg_rmse'].min():.4f}\n")
        f.write(f"Worst RMSE: {summary_df['avg_rmse'].max():.4f}\n")
        f.write(f"Median RMSE: {summary_df['avg_rmse'].median():.4f}\n")
        f.write(f"\n")
        f.write(f"Average NDCG@3: {summary_df['avg_ndcg@3'].mean():.4f} ± {summary_df['avg_ndcg@3'].std():.4f}\n")
        f.write(f"Average NDCG@5: {summary_df['avg_ndcg@5'].mean():.4f} ± {summary_df['avg_ndcg@5'].std():.4f}\n")
        f.write(f"Average NDCG@10: {summary_df['avg_ndcg@10'].mean():.4f} ± {summary_df['avg_ndcg@10'].std():.4f}\n")
        f.write(f"Best NDCG@10: {summary_df['avg_ndcg@10'].max():.4f}\n")
        f.write(f"Worst NDCG@10: {summary_df['avg_ndcg@10'].min():.4f}\n")
        f.write(f"Median NDCG@10: {summary_df['avg_ndcg@10'].median():.4f}\n")
        f.write(f"\nFixed parameters:\n")
        params = get_fixed_params()
        for key, value in params.items():
            f.write(f"  {key}: {value}\n")
    
    logger.info(f"Statistics saved to {stats_file}")
    
    return summary_df, detailed_df

def plot_results(summary_df, log_dir):
    """結果をプロットする関数"""
    logger = logging.getLogger(__name__)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('100 Iterations Training Results (MSE Loss with NDCG Evaluation)', fontsize=16)
    
    # 1. イテレーションごとの平均NDCG@10
    axes[0, 0].plot(summary_df['iteration'], summary_df['avg_ndcg@10'], 'b-', alpha=0.7)
    axes[0, 0].fill_between(summary_df['iteration'], 
                           summary_df['avg_ndcg@10'] - summary_df['std_ndcg@10'],
                           summary_df['avg_ndcg@10'] + summary_df['std_ndcg@10'], 
                           alpha=0.3)
    axes[0, 0].set_title('Average NDCG@10 per Iteration')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('NDCG@10')
    axes[0, 0].grid(True)
    
    # 2. NDCG@10の分布
    axes[0, 1].hist(summary_df['avg_ndcg@10'], bins=20, alpha=0.7, color='green')
    axes[0, 1].axvline(summary_df['avg_ndcg@10'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {summary_df["avg_ndcg@10"].mean():.4f}')
    axes[0, 1].set_title('Distribution of Average NDCG@10')
    axes[0, 1].set_xlabel('NDCG@10')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. NDCG@3, @5, @10の比較
    axes[0, 2].plot(summary_df['iteration'], summary_df['avg_ndcg@3'], 'r-', alpha=0.7, label='NDCG@3')
    axes[0, 2].plot(summary_df['iteration'], summary_df['avg_ndcg@5'], 'g-', alpha=0.7, label='NDCG@5')
    axes[0, 2].plot(summary_df['iteration'], summary_df['avg_ndcg@10'], 'b-', alpha=0.7, label='NDCG@10')
    axes[0, 2].set_title('NDCG Comparison (@3, @5, @10)')
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_ylabel('NDCG')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # 4. RMSE推移
    axes[1, 0].plot(summary_df['iteration'], summary_df['avg_rmse'], 'orange', alpha=0.7)
    axes[1, 0].fill_between(summary_df['iteration'], 
                           summary_df['avg_rmse'] - summary_df['std_rmse'],
                           summary_df['avg_rmse'] + summary_df['std_rmse'], 
                           alpha=0.3, color='orange')
    axes[1, 0].set_title('Average RMSE per Iteration')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].grid(True)
    
    # 5. RMSE vs NDCG@10の散布図
    axes[1, 1].scatter(summary_df['avg_rmse'], summary_df['avg_ndcg@10'], alpha=0.6, color='purple')
    axes[1, 1].set_title('RMSE vs NDCG@10')
    axes[1, 1].set_xlabel('RMSE')
    axes[1, 1].set_ylabel('NDCG@10')
    axes[1, 1].grid(True)
    
    # 相関係数を表示
    correlation = summary_df['avg_rmse'].corr(summary_df['avg_ndcg@10'])
    axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=axes[1, 1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 6. 標準偏差の推移（NDCG@10）
    axes[1, 2].plot(summary_df['iteration'], summary_df['std_ndcg@10'], 'purple', marker='o', markersize=2)
    axes[1, 2].set_title('Standard Deviation of NDCG@10 per Iteration')
    axes[1, 2].set_xlabel('Iteration')
    axes[1, 2].set_ylabel('Standard Deviation')
    axes[1, 2].grid(True)
    
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
    rmse_scores = [r['avg_rmse'] for r in successful_results]
    ndcg3_scores = [r['avg_ndcg@3'] for r in successful_results]
    ndcg5_scores = [r['avg_ndcg@5'] for r in successful_results]
    ndcg10_scores = [r['avg_ndcg@10'] for r in successful_results]
    
    # 最終統計をTensorBoardに記録
    tb_writer.add_scalar("final_statistics/overall_mean_rmse", np.mean(rmse_scores), 0)
    tb_writer.add_scalar("final_statistics/overall_std_rmse", np.std(rmse_scores), 0)
    tb_writer.add_scalar("final_statistics/overall_min_rmse", np.min(rmse_scores), 0)
    tb_writer.add_scalar("final_statistics/overall_max_rmse", np.max(rmse_scores), 0)
    tb_writer.add_scalar("final_statistics/overall_median_rmse", np.median(rmse_scores), 0)
    
    tb_writer.add_scalar("final_statistics/overall_mean_ndcg@3", np.mean(ndcg3_scores), 0)
    tb_writer.add_scalar("final_statistics/overall_mean_ndcg@5", np.mean(ndcg5_scores), 0)
    tb_writer.add_scalar("final_statistics/overall_mean_ndcg@10", np.mean(ndcg10_scores), 0)
    tb_writer.add_scalar("final_statistics/overall_std_ndcg@10", np.std(ndcg10_scores), 0)
    tb_writer.add_scalar("final_statistics/overall_max_ndcg@10", np.max(ndcg10_scores), 0)
    tb_writer.add_scalar("final_statistics/overall_min_ndcg@10", np.min(ndcg10_scores), 0)
    tb_writer.add_scalar("final_statistics/overall_median_ndcg@10", np.median(ndcg10_scores), 0)
    
    tb_writer.add_scalar("final_statistics/successful_iterations", len(successful_results), 0)
    
    # ヒストグラムも追加
    tb_writer.add_histogram("final_statistics/rmse_distribution", np.array(rmse_scores), 0)
    tb_writer.add_histogram("final_statistics/ndcg@10_distribution", np.array(ndcg10_scores), 0)

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
        
        logger.info("Starting 100 iterations training with MSE loss and NDCG evaluation...")
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
                    rmse_scores = [r['avg_rmse'] for r in completed_results]
                    ndcg10_scores = [r['avg_ndcg@10'] for r in completed_results]
                    current_rmse_mean = np.mean(rmse_scores)
                    current_rmse_std = np.std(rmse_scores)
                    current_ndcg10_mean = np.mean(ndcg10_scores)
                    current_ndcg10_std = np.std(ndcg10_scores)
                    logger.info(f"Progress: {i+1}/100 - RMSE: {current_rmse_mean:.4f}±{current_rmse_std:.4f}, NDCG@10: {current_ndcg10_mean:.4f}±{current_ndcg10_std:.4f}")
                    
                    # TensorBoardに中間統計を記録
                    if tb_writer is not None:
                        tb_writer.add_scalar("progress/mean_rmse", current_rmse_mean, i+1)
                        tb_writer.add_scalar("progress/std_rmse", current_rmse_std, i+1)
                        tb_writer.add_scalar("progress/mean_ndcg@10", current_ndcg10_mean, i+1)
                        tb_writer.add_scalar("progress/std_ndcg@10", current_ndcg10_std, i+1)
        
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
        rmse_scores = [r['avg_rmse'] for r in successful_results]
        ndcg3_scores = [r['avg_ndcg@3'] for r in successful_results]
        ndcg5_scores = [r['avg_ndcg@5'] for r in successful_results]
        ndcg10_scores = [r['avg_ndcg@10'] for r in successful_results]
        
        print("\n" + "="*70)
        print("100 ITERATIONS TRAINING COMPLETED (MSE Loss with NDCG Evaluation)")
        print("="*70)
        print(f"Successful iterations: {len(successful_results)}/100")
        print(f"\nRMSE Results:")
        print(f"  Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
        print(f"  Best RMSE: {np.min(rmse_scores):.4f}")
        print(f"  Worst RMSE: {np.max(rmse_scores):.4f}")
        print(f"  Median RMSE: {np.median(rmse_scores):.4f}")
        print(f"\nNDCG Results:")
        print(f"  Average NDCG@3: {np.mean(ndcg3_scores):.4f} ± {np.std(ndcg3_scores):.4f}")
        print(f"  Average NDCG@5: {np.mean(ndcg5_scores):.4f} ± {np.std(ndcg5_scores):.4f}")
        print(f"  Average NDCG@10: {np.mean(ndcg10_scores):.4f} ± {np.std(ndcg10_scores):.4f}")
        print(f"  Best NDCG@10: {np.max(ndcg10_scores):.4f}")
        print(f"  Worst NDCG@10: {np.min(ndcg10_scores):.4f}")
        print(f"  Median NDCG@10: {np.median(ndcg10_scores):.4f}")
        print(f"\nResults saved in: {log_dir}")
        
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