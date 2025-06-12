import hashlib
import os

# MSLRデータのディレクトリ
data_dir = "/workspace/data/raw/mslr-web"
folds = ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5"]
file_types = ["train.txt", "vali.txt", "test.txt"]

print(f"MSLRデータセット分析: {data_dir}")
print("=" * 50)

# 各フォルドのファイル情報収集
all_files = []
for fold in folds:
    for file_type in file_types:
        file_path = os.path.join(data_dir, fold, file_type)
        if os.path.exists(file_path):
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            all_files.append({
                "fold": fold,
                "type": file_type,
                "path": file_path,
                "size_mb": file_size_mb
            })

# ファイルサイズの合計を計算
total_size_mb = sum(f["size_mb"] for f in all_files)
print(f"合計ファイル数: {len(all_files)}")
print(f"合計サイズ: {total_size_mb:.2f} MB")

# 最初に各ファイルの次元数を確認
for file_info in all_files[:1]:  # 最初のファイルだけチェック
    with open(file_info["path"], 'r') as f:
        first_line = f.readline().strip()
        parts = first_line.split()
        feature_ids = [int(part.split(':')[0]) for part in parts[2:]]
        
        print(f"\n特徴量の次元数: {max(feature_ids)}")
        print(f"特徴量の種類数: {len(set(feature_ids))}")
        print(f"特徴量ID: {sorted(set(feature_ids))[:10]}... (最初の10個のみ表示)")

# すべてのファイルを合わせた一意の特徴ベクトル数を計算
print("\nすべてのデータの一意特徴ベクトル数を計算中...")
unique_vectors = set()
total_vectors = 0
unique_qids = set()

for i, file_info in enumerate(all_files):
    print(f"\n処理中 [{i+1}/{len(all_files)}]: {file_info['fold']}/{file_info['type']}")
    file_vectors = 0
    
    with open(file_info["path"], 'r') as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split()
            qid = parts[1].split(":")[1]
            unique_qids.add(qid)
            
            # 特徴ベクトルをハッシュ化
            features = " ".join(parts[2:])
            fhash = hashlib.md5(features.encode()).hexdigest()
            unique_vectors.add(fhash)
            
            total_vectors += 1
            file_vectors += 1
            
            # 進捗表示
            if line_idx > 0 and line_idx % 200000 == 0:
                print(f"  {line_idx}行処理、現在の一意ベクトル数: {len(unique_vectors)}")
    
    print(f"  {file_info['fold']}/{file_info['type']} 完了: {file_vectors}行")
    
    # 途中経過の表示
    unique_ratio = (len(unique_vectors) / total_vectors) * 100
    print(f"  現在の合計: {total_vectors}行、一意ベクトル数: {len(unique_vectors)}、一意率: {unique_ratio:.2f}%")

# 最終結果
print("\n" + "=" * 50)
print("最終結果:")
print(f"処理した合計行数: {total_vectors}")
print(f"一意の特徴ベクトル数: {len(unique_vectors)}")
print(f"一意率: {(len(unique_vectors) / total_vectors) * 100:.2f}%")
print(f"一意のクエリID数: {len(unique_qids)}")
print("=" * 50) 