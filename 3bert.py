import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import pandas as pd
import os
import time
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"=" * 60)
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"=" * 60)

# ==================== 1. 数据加载 ====================
print("\n" + "=" * 60)
print("步骤1: 加载数据")
print("=" * 60)

data_dir = './data'
positive_path = os.path.join(data_dir, 'positive_trainingSet')
negative_path = os.path.join(data_dir, 'negative_trainingSet')
test_path = os.path.join(data_dir, 'testSet-1000_cleaned.xlsx')

# 读取训练数据
positive_texts = []
negative_texts = []

print("读取正例训练集...")
if os.path.isfile(positive_path):
    with open(positive_path, 'r', encoding='utf-8', errors='ignore') as f:
        positive_texts = [line.strip() for line in f if line.strip()]
elif os.path.isdir(positive_path):
    for file in os.listdir(positive_path):
        if file.endswith('.txt'):
            with open(os.path.join(positive_path, file), 'r', encoding='utf-8', errors='ignore') as f:
                positive_texts.extend([line.strip() for line in f if line.strip()])

print(f"  正例: {len(positive_texts)}条")

print("读取负例训练集...")
if os.path.isfile(negative_path):
    with open(negative_path, 'r', encoding='utf-8', errors='ignore') as f:
        negative_texts = [line.strip() for line in f if line.strip()]
elif os.path.isdir(negative_path):
    for file in os.listdir(negative_path):
        if file.endswith('.txt'):
            with open(os.path.join(negative_path, file), 'r', encoding='utf-8', errors='ignore') as f:
                negative_texts.extend([line.strip() for line in f if line.strip()])

print(f"  负例: {len(negative_texts)}条")

# 读取测试集
print("读取测试集...")
test_data = pd.read_excel(test_path)
print(f"测试集列名: {test_data.columns.tolist()}")

test_texts = []
test_labels = []

for col in test_data.columns:
    if 'machine' in col.lower() or 'title given' in col.lower():
        test_texts = test_data[col].astype(str).tolist()
    if col in ['Y/N', 'YN', 'Label', 'label']:
        labels_raw = test_data[col].astype(str).str.upper().tolist()
        test_labels = [1 if l == 'Y' else 0 for l in labels_raw]

print(f"  测试集: {len(test_texts)}条 (正确:{sum(test_labels)}, 错误:{len(test_labels)-sum(test_labels)})")

# 构建完整训练集
all_train_texts = positive_texts + negative_texts
all_train_labels = [1] * len(positive_texts) + [0] * len(negative_texts)

print(f"\n总训练数据: {len(all_train_texts)}条")

# ==================== 2. 划分训练集和验证集 ====================
print("\n" + "=" * 60)
print("步骤2: 划分训练集和验证集 (8:2)")
print("=" * 60)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    all_train_texts, all_train_labels,
    test_size=0.2,
    random_state=42,
    stratify=all_train_labels
)

print(f"训练集: {len(train_texts)}条")
print(f"验证集: {len(val_texts)}条")
print(f"测试集: {len(test_texts)}条")

# ==================== 3. 加载BERT模型 ====================
print("\n" + "=" * 60)
print("步骤3: 加载BERT模型")
print("=" * 60)

model_path = './bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(
    model_path,
    num_labels=2,
    output_hidden_states=True
).to(device)

print(f"模型加载成功")
print(f"Transformer层数: {model.config.num_hidden_layers}")

# ==================== 4. 创建数据集 ====================
print("\n" + "=" * 60)
print("步骤4: 创建数据集")
print("=" * 60)

class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

batch_size = 32
train_dataset = BertDataset(train_texts, train_labels, tokenizer)
val_dataset = BertDataset(val_texts, val_labels, tokenizer)
test_dataset = BertDataset(test_texts, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练批次数: {len(train_loader)}")
print(f"验证批次数: {len(val_loader)}")
print(f"测试批次数: {len(test_loader)}")

# ==================== 5. 训练 ====================
print("\n" + "=" * 60)
print("步骤5: 开始微调训练")
print("=" * 60)

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

from torch.optim.lr_scheduler import LinearLR
total_steps = len(train_loader) * 5
scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=total_steps // 10)

print("开始训练...")
start_time = time.time()

train_losses = []
val_accuracies = []
val_f1_scores = []
best_val_f1 = 0

for epoch in range(5):
    print(f"\nEpoch {epoch + 1}/5")
    print("-" * 40)
    
    # 训练阶段
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if batch_idx % 500 == 0:
            print(f"  批次 {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"  平均训练损失: {avg_loss:.4f}")
    
    # 验证阶段
    model.eval()
    val_preds = []
    val_labels_list = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels_list.extend(labels)
    
    val_acc = accuracy_score(val_labels_list, val_preds)
    val_f1 = precision_recall_fscore_support(val_labels_list, val_preds, average='binary')[2]
    val_accuracies.append(val_acc)
    val_f1_scores.append(val_f1)
    
    print(f"  验证准确率: {val_acc:.4f}")
    print(f"  验证F1分数: {val_f1:.4f}")
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"  → 保存最佳模型 (F1={val_f1:.4f})")

train_time = time.time() - start_time
print(f"\n训练完成！总耗时: {train_time:.2f}秒 ({train_time/60:.2f}分钟)")

# ==================== 6. 最终测试评估 ====================
print("\n" + "=" * 60)
print("步骤6: 最终测试评估")
print("=" * 60)

model.load_state_dict(torch.load('best_model.pt'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].cpu().numpy()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

final_acc = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
cm = confusion_matrix(all_labels, all_preds)

print(f"\n测试集评估结果:")
print(f"  准确率: {final_acc:.4f}")
print(f"  F1分数: {f1:.4f}")
print(f"  精确率: {precision:.4f}")
print(f"  召回率: {recall:.4f}")
print(f"\n混淆矩阵:")
print(cm)

# ==================== 7. 提取12个Transformer层特征 ====================
print("\n" + "=" * 60)
print("步骤7: 提取12个Transformer层特征（用于T-SNE）")
print("=" * 60)

def extract_transformer_layers_features(model, dataloader, device):
    """提取全部12个Transformer层的[CLS] token特征"""
    num_layers = model.config.num_hidden_layers
    all_features = [[] for _ in range(num_layers)]
    all_labels = []
    
    print(f"提取 {len(dataloader.dataset)} 条测试集的特征...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Transformer各层: hidden_states[0]到[11]
            for layer_idx in range(num_layers):
                cls_features = outputs.hidden_states[layer_idx][:, 0, :].cpu().numpy()
                all_features[layer_idx].append(cls_features)
            
            all_labels.extend(labels)
            
            if batch_idx % 20 == 0:
                print(f"  已处理 {batch_idx * len(input_ids)}/{len(dataloader.dataset)} 条")
    
    for layer_idx in range(num_layers):
        all_features[layer_idx] = np.vstack(all_features[layer_idx])
    
    return np.array(all_labels), all_features

print("提取测试集12个Transformer层特征...")
start_time = time.time()
test_labels_arr, test_features_all_layers = extract_transformer_layers_features(model, test_loader, device)
print(f"特征提取耗时: {time.time() - start_time:.2f}秒")

# 验证特征维度
num_layers = model.config.num_hidden_layers
print(f"\n验证特征维度 (12个Transformer层):")
for layer_idx in range(num_layers):
    print(f"  Layer {layer_idx}: {test_features_all_layers[layer_idx].shape}")

# ==================== 8. T-SNE可视化全部12个Transformer层 ====================
print("\n" + "=" * 60)
print("步骤8: T-SNE可视化全部12个Transformer层 (3x4布局)")
print("=" * 60)

total_samples = len(test_labels_arr)
print(f"使用全部 {total_samples} 个测试集样本进行T-SNE可视化")
print("注意：T-SNE计算量较大，可能需要较长时间，请耐心等待...")

# 创建子图：3行4列（正好12个）
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

for layer_idx in range(num_layers):
    print(f"\n处理 Layer {layer_idx} 的T-SNE...")
    start = time.time()
    
    X_layer = test_features_all_layers[layer_idx]
    y_layer = test_labels_arr
    
    # T-SNE降维
    print(f"  正在执行T-SNE降维（{X_layer.shape[0]}个样本，{X_layer.shape[1]}维）...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, X_layer.shape[0] - 1),
        learning_rate=200,
        n_iter=1000,
        method='barnes_hut',
        n_jobs=24,
        verbose=0
    )
    X_tsne = tsne.fit_transform(X_layer)
    
    # 绘图
    ax = axes[layer_idx]
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                         c=y_layer, cmap='RdYlGn', 
                         alpha=0.6, s=5, edgecolors='none')
    
    # 设置英文标题和标签
    ax.set_title(f'Transformer Layer {layer_idx}', fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 在每个子图右下角添加小图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#d73027', alpha=0.6, label='Wrong Title (0)'),
                       Patch(facecolor='#1a9850', alpha=0.6, label='Correct Title (1)')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.7)
    
    elapsed = time.time() - start
    print(f"  完成，耗时: {elapsed:.2f}秒")

# 添加整体标题
plt.suptitle(f't-SNE Visualization of 12 Transformer Layers\nFine-tuned BERT | Test Accuracy: {final_acc:.4f}', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('tsne_all_12_transformer_layers.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n全部12层T-SNE可视化已保存: tsne_all_12_transformer_layers.png")

# ==================== 9. 训练曲线可视化 ====================
print("\n" + "=" * 60)
print("步骤9: 训练曲线可视化")
print("=" * 60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 训练损失曲线
ax1.plot(range(1, 6), train_losses, 'b-o', linewidth=2, markersize=8, label='Training Loss')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training Loss over Epochs', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 验证指标曲线
ax2.plot(range(1, 6), val_accuracies, 'g-o', linewidth=2, markersize=8, label='Validation Accuracy')
ax2.plot(range(1, 6), val_f1_scores, 'r-s', linewidth=2, markersize=8, label='Validation F1 Score')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Validation Metrics over Epochs', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.5, 1.0])

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"训练曲线已保存: training_curves.png")

# ==================== 10. 保存微调后的模型 ====================
print("\n" + "=" * 60)
print("步骤10: 保存微调后的模型")
print("=" * 60)

model_save_path = './bert_finetuned_model'
os.makedirs(model_save_path, exist_ok=True)
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"微调后的模型已保存到: {model_save_path}")

# ==================== 11. 总结 ====================
print("\n" + "=" * 60)
print("实验完成！")
print("=" * 60)
print(f"\n生成的文件:")
print("  1. training_curves.png - 训练损失和验证曲线")
print("  2. tsne_all_12_transformer_layers.png - 全部12个Transformer层T-SNE可视化 (3x4布局)")
print("  3. bert_finetuned_model/ - 微调后的模型")
print(f"\n最终结果:")
print(f"  BERT微调测试准确率: {final_acc:.4f}")
print(f"  BERT微调测试F1分数: {f1:.4f}")
print(f"  最佳验证F1分数: {best_val_f1:.4f}")
print(f"\n实验配置:")
print(f"  设备: {device}")
print(f"  批次大小: {batch_size}")
print(f"  学习率: 2e-5")
print(f"  训练轮数: 5")
print(f"  训练/验证划分: 8:2")
print("=" * 60)