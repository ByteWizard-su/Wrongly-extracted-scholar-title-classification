import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from datetime import datetime

print("=" * 60)
print("朴素贝叶斯分类器 - 错误标题检测")
print("=" * 60)

# ==================== 1. 数据加载 ====================
print("\n" + "=" * 60)
print("步骤1: 加载数据")
print("=" * 60)

def load_title_file(file_path):
    """从文本文件加载标题，假设每行一个标题"""
    titles = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                title = line.strip()
                if title:
                    titles.append(title)
    except Exception as e:
        print(f"  加载 {file_path} 时出错: {e}")
        titles = []
    return titles

# 使用与BERT相同的数据路径
data_dir = './data'
positive_path = os.path.join(data_dir, 'positive_trainingSet')
negative_path = os.path.join(data_dir, 'negative_trainingSet')
test_path = os.path.join(data_dir, 'testSet-1000_cleaned.xlsx')

# 加载正样本
print("读取正例训练集...")
train_titles_positive = load_title_file(positive_path)
print(f"  正例: {len(train_titles_positive)}条")

# 加载负样本
print("读取负例训练集...")
train_titles_negative = load_title_file(negative_path)
print(f"  负例: {len(train_titles_negative)}条")

# 加载测试集
print("读取测试集...")
test_df = pd.read_excel(test_path)
print(f"测试集列名: {test_df.columns.tolist()}")

# 识别测试集列
test_titles = []
test_labels = []

for col in test_df.columns:
    if 'machine' in col.lower() or 'title given' in col.lower():
        test_titles = test_df[col].astype(str).tolist()
    if col in ['Y/N', 'YN', 'Label', 'label']:
        labels_raw = test_df[col].astype(str).str.upper().tolist()
        test_labels = [1 if l == 'Y' else 0 for l in labels_raw]

print(f"测试集: {len(test_titles)}条 (正确:{sum(test_labels)}, 错误:{len(test_labels)-sum(test_labels)})")

# 合并训练集
train_titles = train_titles_positive + train_titles_negative
train_labels = [1] * len(train_titles_positive) + [0] * len(train_titles_negative)

print(f"\n训练集总计: {len(train_titles)}条")
print(f"  正例: {len(train_titles_positive)}条")
print(f"  负例: {len(train_titles_negative)}条")

# ==================== 2. 特征提取 ====================
print("\n" + "=" * 60)
print("步骤2: 特征提取")
print("=" * 60)

vectorizer_final = TfidfVectorizer(
    max_features=20000,
    min_df=2,
    max_df=0.8,
    stop_words='english',
    ngram_range=(1, 3),
    sublinear_tf=True,
    norm='l2'
)

X_train_final = vectorizer_final.fit_transform(train_titles)
X_test_final = vectorizer_final.transform(test_titles)

print(f"特征维度: {X_train_final.shape[1]}")
print(f"训练集特征矩阵: {X_train_final.shape}")
print(f"测试集特征矩阵: {X_test_final.shape}")

# ==================== 3. 训练朴素贝叶斯模型 ====================
print("\n" + "=" * 60)
print("步骤3: 训练朴素贝叶斯模型")
print("=" * 60)

# 设置类别权重
final_weights = [1.05, 0.95]
final_alpha = 8.0
sample_weights_final = np.array([final_weights[label] for label in train_labels])

# 训练模型
base_model = MultinomialNB(alpha=final_alpha)
base_model.fit(X_train_final, train_labels, sample_weight=sample_weights_final)

# 预测
base_predictions = base_model.predict(X_test_final)
base_probabilities = base_model.predict_proba(X_test_final)
base_accuracy = accuracy_score(test_labels, base_predictions)

print("模型训练完成！")

# ==================== 4. 模型评估 ====================
print("\n" + "=" * 60)
print("步骤4: 模型评估")
print("=" * 60)

# 计算评估指标
acc = accuracy_score(test_labels, base_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(test_labels, base_predictions, average='binary')
cm = confusion_matrix(test_labels, base_predictions)

print(f"\n测试集评估结果:")
print(f"  准确率: {acc:.4f}")
print(f"  F1分数: {f1:.4f}")
print(f"  精确率: {precision:.4f}")
print(f"  召回率: {recall:.4f}")
print(f"\n混淆矩阵:")
print(cm)

# ==================== 5. T-SNE可视化 ====================
print("\n" + "=" * 60)
print("步骤5: T-SNE可视化")
print("=" * 60)

# 使用TF-IDF特征进行T-SNE降维（采样以避免内存问题）
sample_size = min(1000, X_test_final.shape[0])
sample_indices = np.random.choice(X_test_final.shape[0], sample_size, replace=False)

# 提取采样特征
X_test_sample = X_test_final[sample_indices].toarray()
y_test_sample = np.array(test_labels)[sample_indices]

print(f"使用 {sample_size} 个测试集样本进行T-SNE可视化")
print("注意：T-SNE计算量较大，可能需要一些时间，请耐心等待...")

print("\n执行T-SNE降维...")
start_time = datetime.now()

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=min(30, sample_size - 1),
    learning_rate=200,
    n_iter=1000,
    method='barnes_hut',
    n_jobs=-1,
    verbose=0
)

X_tsne = tsne.fit_transform(X_test_sample)
print(f"T-SNE完成，耗时: {(datetime.now() - start_time).seconds}秒")

# 绘图
plt.figure(figsize=(12, 10))

scatter = plt.scatter(
    X_tsne[:, 0], X_tsne[:, 1],
    c=y_test_sample, cmap='RdYlGn',
    alpha=0.6, s=15, edgecolors='none'
)

plt.title(f't-SNE Visualization of TF-IDF Features (Naive Bayes)\nTest Accuracy: {acc:.4f}',
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
plt.savefig('tsne_naive_bayes.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nT-SNE可视化已保存: tsne_naive_bayes.png")

# ==================== 6. 保存结果 ====================
print("\n" + "=" * 60)
print("步骤6: 保存结果")
print("=" * 60)

# 创建结果DataFrame
results_df = pd.DataFrame({
    'id': test_df['id'] if 'id' in test_df.columns else range(1, len(test_titles) + 1),
    'title_given_by_machine': test_titles,
    'true_label': ['Y' if l == 1 else 'N' for l in test_labels],
    'predicted_label': ['Y' if p == 1 else 'N' for p in base_predictions],
    'confidence': [max(prob) for prob in base_probabilities],
    'correct': ['✓' if t == p else '✗' for t, p in zip(test_labels, base_predictions)]
})

results_df.to_excel('nb_classification_results.xlsx', index=False)
print(f"预测结果已保存: nb_classification_results.xlsx")

# 统计
correct_count = sum(results_df['correct'] == '✓')
print(f"\n预测正确: {correct_count}/{len(results_df)} ({correct_count/len(results_df)*100:.1f}%)")

# ==================== 7. 总结 ====================
print("\n" + "=" * 60)
print("实验完成！")
print("=" * 60)
print(f"\n生成的文件:")
print("  1. tsne_naive_bayes.png - TF-IDF特征T-SNE可视化")
print("  2. nb_classification_results.xlsx - 测试集预测结果")
print(f"\n最终结果:")
print(f"  朴素贝叶斯测试准确率: {acc:.4f}")
print(f"  朴素贝叶斯测试F1分数: {f1:.4f}")
print(f"\n实验配置:")
print(f"  TF-IDF特征数: {X_train_final.shape[1]}")
print(f"  朴素贝叶斯alpha: {final_alpha}")
print(f"  类别权重: {final_weights}")
print("=" * 60)