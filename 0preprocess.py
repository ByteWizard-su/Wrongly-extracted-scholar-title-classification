import pandas as pd
import numpy as np
import os

def clean_titles(titles_df, title_column_name):
    """
    清洗标题数据，只删除空值（None、NaN、空字符串）
    """
    print(f"原始数据行数: {len(titles_df)}")
    
    # 复制数据框以避免修改原始数据
    cleaned_df = titles_df.copy()
    
    # 统计原始数据中的空值情况
    original_nulls = cleaned_df[title_column_name].isna().sum()
    print(f"原始数据中的NaN/None数量: {original_nulls}")
    
    # 删除包含空值的行
    cleaned_df = cleaned_df.dropna(subset=[title_column_name])
    after_dropna = len(cleaned_df)
    print(f"删除NaN/None后剩余: {after_dropna} (删除了 {original_nulls} 行)")
    
    # 统计并删除空字符串
    empty_strings = (cleaned_df[title_column_name].astype(str).str.strip() == '').sum()
    if empty_strings > 0:
        cleaned_df = cleaned_df[cleaned_df[title_column_name].astype(str).str.strip() != '']
        print(f"删除空字符串后剩余: {len(cleaned_df)} (删除了 {empty_strings} 行)")
    
    # 注意：不过滤过短的标题，只删除空值
    # 只去除标题首尾空格
    cleaned_df[title_column_name] = cleaned_df[title_column_name].astype(str).str.strip()
    
    print(f"\n数据清洗完成！")
    print(f"清洗后数据行数: {len(cleaned_df)}")
    print(f"共删除了 {len(titles_df) - len(cleaned_df)} 行无效数据（仅删除空值）")
    
    return cleaned_df

def clean_testset():
    """
    清洗测试集数据并保存
    """
    print("="*60)
    print("测试集数据清洗（只删除空值）")
    print("="*60)
    
    # 设置文件路径
    data_dir = './data'
    test_file = f'{data_dir}/testSet-1000.xlsx'
    
    # 检查文件是否存在
    if not os.path.exists(test_file):
        print(f"错误：找不到文件 {test_file}")
        return None
    
    # 加载测试集
    print(f"正在加载文件: {test_file}")
    test_df = pd.read_excel(test_file)
    
    # 显示原始数据信息
    print("\n原始测试集信息:")
    print(f"  总行数: {len(test_df)}")
    print(f"  列名: {test_df.columns.tolist()}")
    
    # 检查是否存在'Y/N'列中的空值
    if 'Y/N' in test_df.columns:
        y_n_nulls = test_df['Y/N'].isna().sum()
        if y_n_nulls > 0:
            print(f"警告：'Y/N'列有 {y_n_nulls} 个空值")
            # 删除'Y/N'列为空的行
            test_df = test_df.dropna(subset=['Y/N'])
            print(f"删除'Y/N'列为空后剩余: {len(test_df)} 行")
    
    # 清洗标题数据（标题列名为'title given by manchine'）
    title_column = 'title given by manchine'
    print(f"\n开始清洗标题列: '{title_column}'")
    cleaned_test_df = clean_titles(test_df, title_column)
    
    # 保存清洗后的数据
    output_file = f'{data_dir}/testSet-1000_cleaned.xlsx'
    cleaned_test_df.to_excel(output_file, index=False)
    print(f"\n清洗后的数据已保存到: {output_file}")
    
    # 也可保存为CSV格式作为备份
    output_csv = f'{data_dir}/testSet-1000_cleaned.csv'
    cleaned_test_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"CSV备份已保存到: {output_csv}")
    
    # 显示清洗后的数据统计
    print("\n" + "="*60)
    print("清洗后测试集统计信息")
    print("="*60)
    print(f"  总样本数: {len(cleaned_test_df)}")
    print(f"  正例(Y): {(cleaned_test_df['Y/N'] == 'Y').sum()}")
    print(f"  负例(N): {(cleaned_test_df['Y/N'] == 'N').sum()}")
    print(f"  列名: {cleaned_test_df.columns.tolist()}")
    
    # 显示一些标题示例（包括可能很短的标题）
    print("\n清洗后标题示例（包含短标题）:")
    for i, title in enumerate(cleaned_test_df[title_column].head(10)):
        print(f"  {i+1}. '{title}' (长度: {len(str(title))})")
    
    # 显示最短的几个标题
    title_lengths = cleaned_test_df[title_column].astype(str).str.len()
    print("\n最短的5个标题:")
    shortest_indices = title_lengths.nsmallest(5).index
    for idx in shortest_indices:
        title = cleaned_test_df.loc[idx, title_column]
        print(f"  - '{title}' (长度: {len(str(title))})")
    
    return cleaned_test_df

# 运行数据清洗
if __name__ == "__main__":
    cleaned_data = clean_testset()
    
    print("\n" + "="*60)
    print("数据清洗脚本执行完毕！")
    print("请确保在运行后续实验代码时使用清洗后的文件: './data/testSet-1000_cleaned.xlsx'")
    print("="*60)