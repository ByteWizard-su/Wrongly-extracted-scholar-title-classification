import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import multiprocessing
import gc
from datetime import datetime
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# 设置设备信息
print("=" * 60)
print(f"Word2Vec + SVM Classifier")
print(f"CPU核心数: {multiprocessing.cpu_count()}")
print("=" * 60)

# ====================== 1. 加载数据 ======================
def load_txt(file_path, label, encoding='utf-8'):
    texts = []
    labels = []
    try:
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
                    labels.append(label)
    except UnicodeDecodeError:
        if encoding != 'latin-1':
            return load_txt(file_path, label, encoding='latin-1')
        elif encoding != 'gbk':
            return load_txt(file_path, label, encoding='gbk')
        else:
            raise Exception(f"无法解码文件 {file_path}")
    return texts, labels

print("\n" + "=" * 60)
print("步骤1: 加载数据")
print("=" * 60)

# 使用与BERT相同的数据路径
data_dir = './data'
positive_path = os.path.join(data_dir, 'positive_trainingSet')
negative_path = os.path.join(data_dir, 'negative_trainingSet')
test_path = os.path.join(data_dir, 'testSet-1000_cleaned.xlsx')

# 读取训练数据
print("读取正例训练集...")
pos_texts, pos_labels = load_txt(positive_path, 1)
print(f"  正例: {len(pos_texts)}条")

print("读取负例训练集...")
neg_texts, neg_labels = load_txt(negative_path, 0)
print(f"  负例: {len(neg_texts)}条")

train_texts = pos_texts + neg_texts
train_labels = pos_labels + neg_labels

print(f"\n训练集总计: {len(train_texts)}条")
print(f"正样本数: {len(pos_texts):,}，负样本数: {len(neg_texts):,}")

# 读取测试集
print("\n读取测试集...")
test_df = pd.read_excel(test_path)
print(f"测试集列名: {test_df.columns.tolist()}")

# 识别测试集列
test_texts = []
test_labels = []

for col in test_df.columns:
    if 'machine' in col.lower() or 'title given' in col.lower():
        test_texts = test_df[col].astype(str).tolist()
    if col in ['Y/N', 'YN', 'Label', 'label']:
        labels_raw = test_df[col].astype(str).str.upper().tolist()
        test_labels = [1 if l == 'Y' else 0 for l in labels_raw]

print(f"测试集: {len(test_texts)}条 (正确:{sum(test_labels)}, 错误:{len(test_labels)-sum(test_labels)})")

# ====================== 2. 分词函数 ======================
def tokenize_simple(texts):
    """简化的分词函数"""
    tokenized = []
    for text in texts:
        text_lower = text.lower()
        tokens = [token for token in text_lower.split() if token]
        tokenized.append(tokens)
    return tokenized

print("\n" + "=" * 60)
print("步骤2: 分词处理")
print("=" * 60)

train_tokens = tokenize_simple(train_texts)
test_tokens = tokenize_simple(test_texts)
print(f"训练集分词完成，样本数: {len(train_tokens):,}")
print(f"测试集分词完成，样本数: {len(test_tokens):,}")

# ====================== 3. 训练Word2Vec模型 ======================
print("\n" + "=" * 60)
print("步骤3: 训练Word2Vec模型")
print("=" * 60)

cpu_cores = max(1, multiprocessing.cpu_count() - 1)

w2v_model = Word2Vec(
    sentences=train_tokens,
    vector_size=300,
    window=8,
    min_count=3,
    workers=cpu_cores,
    sg=1,
    epochs=20,
    seed=42
)

print(f"Word2Vec训练完成！词汇表大小：{len(w2v_model.wv):,}")
print(f"词向量维度: {w2v_model.vector_size}")

# ====================== 4. 训练TF-IDF向量器 ======================
print("\n" + "=" * 60)
print("步骤4: 训练TF-IDF向量器")
print("=" * 60)

tfidf_vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.85
)

# 抽样训练
SAMPLE_TFIDF = min(50000, len(train_texts))
sample_indices = np.random.choice(len(train_texts), SAMPLE_TFIDF, replace=False)
sample_texts = [train_texts[i] for i in sample_indices]

tfidf_vectorizer.fit(sample_texts)
print(f"TF-IDF特征数：{len(tfidf_vectorizer.get_feature_names_out()):,}")

# ====================== 5. 特征提取函数 ======================
def extract_fixed_features(text, tokens, model, tfidf_vec, doc_tfidf=None):
    """特征提取函数，确保固定维度"""
    vector_size = model.vector_size
    FIXED_DIM = vector_size * 3 + 20
    
    features = np.zeros(FIXED_DIM)
    current_idx = 0
    
    # 1. Word2Vec特征（均值、最大值、最小值）
    word_vectors = []
    for word in tokens:
        if word in model.wv:
            word_vectors.append(model.wv[word])
    
    if word_vectors:
        word_vectors = np.array(word_vectors)
        features[current_idx:current_idx+vector_size] = np.mean(word_vectors, axis=0)
        current_idx += vector_size
        features[current_idx:current_idx+vector_size] = np.max(word_vectors, axis=0)
        current_idx += vector_size
        features[current_idx:current_idx+vector_size] = np.min(word_vectors, axis=0)
        current_idx += vector_size
    else:
        current_idx += vector_size * 3
    
    # 2. 文本统计特征
    text_len = len(text)
    num_words = len(tokens)
    
    features[current_idx] = min(text_len / 200.0, 1.0)
    features[current_idx+1] = min(num_words / 30.0, 1.0)
    features[current_idx+2] = sum(1 for c in text if c.isupper()) / max(1, text_len)
    features[current_idx+3] = sum(1 for c in text if c.isdigit()) / max(1, text_len)
    features[current_idx+4] = min(text.count('.') / 10.0, 1.0)
    features[current_idx+5] = min(text.count('-') / 5.0, 1.0)
    features[current_idx+6] = min(text.count(',') / 5.0, 1.0)
    features[current_idx+7] = min(text.count(':') / 3.0, 1.0)
    
    text_lower = text.lower()
    features[current_idx+8] = 1.0 if ('call for paper' in text_lower or 'cfp' in text_lower) else 0.0
    features[current_idx+9] = 1.0 if ('abstract' in text_lower and num_words < 10) else 0.0
    features[current_idx+10] = 1.0 if ('........' in text or '............' in text) else 0.0
    features[current_idx+11] = 1.0 if ('license' in text_lower or 'copyright' in text_lower) else 0.0
    features[current_idx+12] = 1.0 if ('http://' in text_lower or 'https://' in text_lower) else 0.0
    features[current_idx+13] = 1.0 if text.isupper() else 0.0
    features[current_idx+14] = 1.0 if num_words <= 3 else 0.0
    
    current_idx += 15
    
    # 3. TF-IDF加权特征
    if doc_tfidf is not None:
        tfidf_sum = 0
        tfidf_weighted_vec = np.zeros(vector_size)
        
        for word in tokens:
            if word in model.wv and word in tfidf_vec.vocabulary_:
                weight = doc_tfidf[0, tfidf_vec.vocabulary_[word]]
                tfidf_sum += weight
                tfidf_weighted_vec += model.wv[word] * weight
        
        if tfidf_sum > 0:
            tfidf_weighted_vec /= tfidf_sum
        
        remaining_space = FIXED_DIM - current_idx
        if remaining_space >= vector_size:
            features[current_idx:current_idx+vector_size] = tfidf_weighted_vec
    
    return features

# ====================== 6. 处理训练数据 ======================
print("\n" + "=" * 60)
print("步骤5: 提取训练集特征")
print("=" * 60)

print("计算TF-IDF矩阵...")
tfidf_matrix = tfidf_vectorizer.transform(train_texts)

BATCH_SIZE = 20000
num_batches = (len(train_texts) + BATCH_SIZE - 1) // BATCH_SIZE

X_train_batches = []
feature_dim = None

for batch_idx in range(num_batches):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min((batch_idx + 1) * BATCH_SIZE, len(train_texts))
    
    print(f"  处理批次 {batch_idx+1}/{num_batches} ({start_idx:,}-{end_idx:,})")
    
    batch_features = []
    for i in range(start_idx, end_idx):
        features = extract_fixed_features(
            train_texts[i], train_tokens[i], w2v_model,
            tfidf_vectorizer, tfidf_matrix[i] if i < tfidf_matrix.shape[0] else None
        )
        
        if feature_dim is None:
            feature_dim = len(features)
        elif len(features) != feature_dim:
            if len(features) < feature_dim:
                features = np.pad(features, (0, feature_dim - len(features)), 'constant')
            else:
                features = features[:feature_dim]
        
        batch_features.append(features)
    
    X_train_batches.append(np.array(batch_features))
    gc.collect()

X_train = np.vstack(X_train_batches)
y_train = np.array(train_labels)

print(f"\n训练特征矩阵形状: {X_train.shape}")
print(f"特征维度: {X_train.shape[1]}")

# 释放内存
del train_tokens, X_train_batches, tfidf_matrix
gc.collect()

# ====================== 7. 数据标准化 ======================
print("\n" + "=" * 60)
print("步骤6: 数据标准化")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print(f"标准化完成")

# ====================== 8. 训练线性SVM ======================
print("\n" + "=" * 60)
print("步骤7: 训练线性SVM")
print("=" * 60)

PARAM_SAMPLE_SIZE = min(30000, len(X_train_scaled))
param_indices = np.random.choice(len(X_train_scaled), PARAM_SAMPLE_SIZE, replace=False)
X_param = X_train_scaled[param_indices]
y_param = y_train[param_indices]

print(f"使用 {PARAM_SAMPLE_SIZE:,} 个样本进行参数搜索")

param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0],
    'class_weight': [None, 'balanced']
}

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
linear_svm = LinearSVC(dual=False, max_iter=5000, random_state=42)

grid_search = GridSearchCV(
    linear_svm, param_grid, cv=cv,
    scoring='f1_weighted', n_jobs=1, verbose=1
)

grid_search.fit(X_param, y_param)

print(f"\n最优参数: {grid_search.best_params_}")
print(f"交叉验证F1分数: {grid_search.best_score_:.4f}")

print("\n在全量数据上训练最终模型...")
final_svm = LinearSVC(
    dual=False, max_iter=10000, random_state=42,
    **grid_search.best_params_
)

final_svm.fit(X_train_scaled, y_train)
print("模型训练完成！")

# ====================== 9. 处理测试集 ======================
print("\n" + "=" * 60)
print("步骤8: 处理测试集")
print("=" * 60)

print("计算测试集TF-IDF...")
test_tfidf = tfidf_vectorizer.transform(test_texts)

print("提取测试集特征...")
X_test = []
for i in range(len(test_texts)):
    features = extract_fixed_features(
        test_texts[i], test_tokens[i], w2v_model,
        tfidf_vectorizer, test_tfidf[i]
    )
    
    if len(features) != X_train_scaled.shape[1]:
        if len(features) < X_train_scaled.shape[1]:
            features = np.pad(features, (0, X_train_scaled.shape[1] - len(features)), 'constant')
        else:
            features = features[:X_train_scaled.shape[1]]
    
    X_test.append(features)

X_test = np.array(X_test)
X_test_scaled = scaler.transform(X_test)
y_test = np.array(test_labels)

print(f"测试集特征形状: {X_test_scaled.shape}")

# ====================== 10. 模型评估 ======================
print("\n" + "=" * 60)
print("步骤9: 模型评估")
print("=" * 60)

y_pred = final_svm.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
cm = confusion_matrix(y_test, y_pred)

print(f"\n测试集评估结果:")
print(f"  准确率: {acc:.4f}")
print(f"  F1分数: {f1:.4f}")
print(f"  精确率: {precision:.4f}")
print(f"  召回率: {recall:.4f}")
print(f"\n混淆矩阵:")
print(cm)

# ====================== 11. 提取文档向量用于T-SNE ======================
print("\n" + "=" * 60)
print("步骤10: 提取文档向量用于T-SNE可视化")
print("=" * 60)

def extract_document_vectors(texts, tokens_list, model):
    """提取文档向量（词向量均值）用于T-SNE可视化"""
    doc_vectors = []
    
    for tokens in tokens_list:
        word_vectors = []
        for word in tokens:
            if word in model.wv:
                word_vectors.append(model.wv[word])
        
        if word_vectors:
            doc_vec = np.mean(word_vectors, axis=0)
        else:
            doc_vec = np.zeros(model.vector_size)
        
        doc_vectors.append(doc_vec)
    
    return np.array(doc_vectors)

print(f"提取测试集文档向量（{w2v_model.vector_size}维）...")
test_doc_vectors = extract_document_vectors(test_texts, test_tokens, w2v_model)
print(f"文档向量形状: {test_doc_vectors.shape}")

# ====================== 12. T-SNE可视化 ======================
print("\n" + "=" * 60)
print("步骤11: T-SNE可视化")
print("=" * 60)

total_samples = len(test_doc_vectors)
print(f"使用全部 {total_samples} 个测试集样本进行T-SNE可视化")
print("注意：T-SNE计算量较大，可能需要一些时间，请耐心等待...")

print("\n执行T-SNE降维...")
start_time = datetime.now()

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=min(30, total_samples - 1),
    learning_rate=200,
    n_iter=1000,
    method='barnes_hut',
    n_jobs=-1,
    verbose=0
)

X_tsne = tsne.fit_transform(test_doc_vectors)
print(f"T-SNE完成，耗时: {(datetime.now() - start_time).seconds}秒")

# 绘图
plt.figure(figsize=(12, 10))

scatter = plt.scatter(
    X_tsne[:, 0], X_tsne[:, 1],
    c=y_test, cmap='RdYlGn',
    alpha=0.6, s=15, edgecolors='none'
)

plt.title(f't-SNE Visualization of Document Vectors (Word2Vec + SVM)\nTest Accuracy: {acc:.4f}',
          fontsize=14, fontweight='bold')
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.grid(True, alpha=0.3)

# 添加颜色条
cbar = plt.colorbar(scatter)
cbar.set_label('Class', fontsize=10)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Wrong Title (0)', 'Correct Title (1)'])

plt.tight_layout()
plt.savefig('tsne_word2vec_svm.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nT-SNE可视化已保存: tsne_word2vec_svm.png")

# ====================== 13. 保存模型 ======================
print("\n" + "=" * 60)
print("步骤12: 保存模型")
print("=" * 60)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

model_files = {
    'w2v_model': f"w2v_model_{timestamp}.model",
    'tfidf_vectorizer': f"tfidf_vectorizer_{timestamp}.pkl",
    'scaler': f"scaler_{timestamp}.pkl",
    'svm_model': f"svm_model_{timestamp}.pkl",
}

w2v_model.save(model_files['w2v_model'])
joblib.dump(tfidf_vectorizer, model_files['tfidf_vectorizer'])
joblib.dump(scaler, model_files['scaler'])
joblib.dump(final_svm, model_files['svm_model'])

print("模型文件已保存:")
for name, path in model_files.items():
    print(f"  {name}: {path}")

# ====================== 14. 保存结果 ======================
print("\n" + "=" * 60)
print("步骤13: 保存预测结果")
print("=" * 60)

# 创建结果DataFrame
results_df = pd.DataFrame({
    'id': test_df['id'] if 'id' in test_df.columns else range(1, len(test_texts) + 1),
    'title_given_by_machine': test_texts,
    'true_label': ['Y' if l == 1 else 'N' for l in y_test],
    'predicted_label': ['Y' if p == 1 else 'N' for p in y_pred],
    'correct': ['✓' if t == p else '✗' for t, p in zip(y_test, y_pred)]
})

results_df.to_excel('word2vec_svm_results.xlsx', index=False)
print(f"预测结果已保存: word2vec_svm_results.xlsx")

# 统计
correct_count = sum(results_df['correct'] == '✓')
print(f"\n预测正确: {correct_count}/{len(results_df)} ({correct_count/len(results_df)*100:.1f}%)")

# ====================== 15. 总结 ======================
print("\n" + "=" * 60)
print("实验完成！")
print("=" * 60)
print(f"\n生成的文件:")
print("  1. tsne_word2vec_svm.png - Word2Vec文档向量T-SNE可视化")
print("  2. word2vec_svm_results.xlsx - 测试集预测结果")
print("  3. w2v_model_*.model - Word2Vec模型")
print("  4. tfidf_vectorizer_*.pkl - TF-IDF向量器")
print("  5. scaler_*.pkl - 标准化器")
print("  6. svm_model_*.pkl - SVM模型")
print(f"\n最终结果:")
print(f"  Word2Vec+SVM测试准确率: {acc:.4f}")
print(f"  Word2Vec+SVM测试F1分数: {f1:.4f}")
print(f"\n实验配置:")
print(f"  Word2Vec维度: {w2v_model.vector_size}")
print(f"  特征维度: {X_train.shape[1]}")
print(f"  SVM最优参数: {grid_search.best_params_}")
print("=" * 60)