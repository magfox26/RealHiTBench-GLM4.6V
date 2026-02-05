import json
import os

# 指定文件路径
base_path = '/export/home/xwang/liu/RealHiTBench/data/'
input_file = os.path.join(base_path, 'QA_final.json')

# 读取原始文件
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"原始文件总共有 {len(data['queries'])} 条数据")

# 定义分割规则：文件名和起始ID
split_rules = {
    'QA_htmlfinal.json': 577,
    'QA_jsonfinal.json': 805,
    'QA_latexfinal.json': 873,
    'QA_markdownfinal.json': 725
}

# 为每个文件创建数据
for filename, start_id in split_rules.items():
    # 过滤出从start_id开始的数据
    filtered_queries = [q for q in data['queries'] if q['id'] >= start_id]
    
    # 创建新的数据结构
    new_data = {
        'queries': filtered_queries
    }
    
    # 写入新文件到同一路径
    output_file = os.path.join(base_path, filename)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    
    print(f"✓ 已创建 {output_file}: {len(filtered_queries)} 条数据 (从id {start_id}开始)")

# 验证生成的文件
print("\n验证生成的文件：")
print("="*60)

for filename in split_rules.keys():
    output_file = os.path.join(base_path, filename)
    with open(output_file, 'r', encoding='utf-8') as f:
        file_data = json.load(f)
        queries = file_data['queries']
        
        if queries:
            first_id = queries[0]['id']
            last_id = queries[-1]['id']
            print(f"{filename}:")
            print(f"  数据条数: {len(queries)}")
            print(f"  ID范围: {first_id} - {last_id}")
            print()