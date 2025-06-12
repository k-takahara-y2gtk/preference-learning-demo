import hashlib
import os
from collections import defaultdict
import numpy as np

# MSLRデータのディレクトリ
data_dir = "/workspace/data/raw/mslr-web"
sample_file = os.path.join(data_dir, "Fold1", "train.txt")

print(f"MSLRデータセット文書分析: {data_dir}")
print("=" * 50)

# サンプルファイルを使ってデータ構造を調査
print("データ構造の調査...")
with open(sample_file, 'r') as f:
    first_lines = [f.readline().strip() for _ in range(5)]

print("\nデータ形式サンプル（最初の3行）:")
for i, line in enumerate(first_lines[:3]):
    print(f"行 {i+1}: {line[:100]}...")

# 同じクエリIDに対する特徴量の値を調査
# 文書を識別する特徴量（URLや文書ID）を見つけるため
print("\n同じクエリIDに対する特徴量の分析...")
qid_to_features = defaultdict(list)
sample_size = 5000

with open(sample_file, 'r') as f:
    for i, line in enumerate(f):
        if i >= sample_size:
            break
            
        parts = line.strip().split()
        qid = parts[1].split(':')[1]
        
        # 特徴量の値を抽出
        features = {}
        for item in parts[2:]:
            fid, fval = item.split(':')
            features[int(fid)] = float(fval)
            
        qid_to_features[qid].append(features)

# 同じクエリに対する特徴量の変動を分析
print(f"最初の5つのクエリIDに対する特徴量の変動を分析...")

# 文書IDを特定する候補となる特徴量
potential_doc_id_features = []

for qid in list(qid_to_features.keys())[:5]:
    query_features = qid_to_features[qid]
    if len(query_features) <= 1:
        continue
        
    print(f"\nクエリID {qid} の分析 (文書数: {len(query_features)})")
    
    # 各特徴量のユニーク値の数をカウント
    feature_unique_counts = {}
    for fid in range(1, 137):  # 136個の特徴量
        values = [f.get(fid, 0) for f in query_features]
        unique_values = len(set(values))
        variation = unique_values / len(query_features)
        
        # 高いバリエーションを持つ特徴量は文書IDの候補
        if variation > 0.9:  # 90%以上がユニーク
            feature_unique_counts[fid] = unique_values
            
            # 整数のような値を持つ特徴量を特定（文書IDの可能性）
            is_integer_like = all(v.is_integer() for v in values if isinstance(v, float) and not np.isnan(v))
            if is_integer_like:
                potential_doc_id_features.append(fid)
                print(f"  特徴量 {fid}: {unique_values}個のユニーク値 (変動率: {variation:.2f}) - 整数値のみ")
            else:
                print(f"  特徴量 {fid}: {unique_values}個のユニーク値 (変動率: {variation:.2f})")
                
            # 最初の数値を表示
            print(f"    最初の値: {values[:5]}")

# 文書ID候補の特徴量をカウント
unique_potential_features = set(potential_doc_id_features)
print(f"\n文書ID候補の特徴量: {sorted(unique_potential_features)}")

# MSLR-WEB10Kデータセットの文献調査によると、文書URLはデータには含まれておらず
# 各行は単にクエリと文書のペアを表している
print("\nMSLR-WEB10Kデータセット構造の結論:")
print("1. データセットには136個の特徴量があり、そのうちいくつかは文書IDを示す可能性があります")
print("2. 文書のURLや明示的なIDはデータに含まれていないようです")
print("3. 各特徴ベクトルは1つの文書-クエリペアを表しています")

# 全データセットの分析結果からの推定
print("\n全データセットからの文書数推定:")
total_unique_vectors = 1194835  # 前回の分析で得られた値
total_unique_qids = 10000       # 前回の分析で得られた値

print(f"全データセットの一意クエリ数: {total_unique_qids}")
print(f"全データセットの一意特徴ベクトル数: {total_unique_vectors}")
print(f"クエリあたりの平均一意文書数（推定）: {total_unique_vectors / total_unique_qids:.2f}")

# 文書数の推定
# 注意: MSLR-WEB10Kは各クエリにつき平均120の文書がある（論文より）
# 「文書」とは正確には各クエリに対するURLを表す
print("\nMSLR-WEB10Kデータセットに関する文献情報:")
print("- 各クエリには平均約120の文書（URL）が関連付けられています")
print("- 同じURLが異なるクエリに関連付けられることがあります")
print("- クエリ数は10,000個、一意な特徴ベクトル数は約1,195,000個")

print("\n可能性のある文書数の推定:")
for avg_q in [1, 2, 5, 10]:
    est = total_unique_vectors / avg_q
    print(f"- 各文書が平均{avg_q}個のクエリに現れる場合: {est:.0f}文書")

print("\nMicrosoft Research Web10Kデータセットの公式情報によると、10,000のクエリと約1.2M件のジャッジメントが含まれています")
print("各ジャッジメントは1つのクエリ-URL（文書）ペアに対応するため、実際の文書数は全特徴ベクトル数よりも少ない可能性が高いです")
print("=" * 50) 