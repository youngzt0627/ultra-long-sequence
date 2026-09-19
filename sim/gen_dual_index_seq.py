import csv
import os
from collections import defaultdict

def main():
    data_dir = "data"
    sim_dir = "sim"
    input_path = os.path.join(data_dir, "train_joined_logs.csv")
    
    # 1. 定义输出文件路径
    output_files = {
        "item_id": os.path.join(sim_dir, "userId_cateId2itemId.txt"),
        "cate_id": os.path.join(sim_dir, "userId_cateId2cateId.txt"),
        "brand": os.path.join(sim_dir, "userId_cateId2brand.txt"),
        "u_cate_click_cnt": os.path.join(sim_dir, "userId_cateId2uCateClickCnt.txt"),
        "u_brand_click_cnt": os.path.join(sim_dir, "userId_cateId2uBrandClickCnt.txt")
    }
    
    # 2. 建立双索引字典：(userId, cateId) -> [item1, item2, ...]
    dual_index_seqs = {
        "item_id": defaultdict(list),
        "cate_id": defaultdict(list),
        "brand": defaultdict(list),
        "u_cate_click_cnt": defaultdict(list),
        "u_brand_click_cnt": defaultdict(list)
    }
    
    print(f"正在读取 {input_path} 并建立双索引序列...")
    if not os.path.exists(input_path):
        print(f"错误: 找不到输入文件 {input_path}，请先运行 build_joined_logs.py。")
        return

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            # 仅处理点击行为 (clk=1)
            if row.get("clk") == "1":
                uid = row["user_id"]
                cid = row["cate_id"]
                
                # 构建 Key: userId_cateId
                key = f"{uid}_{cid}"
                
                # 提取完整序列
                dual_index_seqs["item_id"][key].append(row["item_id"])
                dual_index_seqs["cate_id"][key].append(row["cate_id"])
                dual_index_seqs["brand"][key].append(row["brand"] or "NULL")
                # 提取交叉特征序列
                dual_index_seqs["u_cate_click_cnt"][key].append(row.get("u_cate_click_cnt", "0"))
                dual_index_seqs["u_brand_click_cnt"][key].append(row.get("u_brand_click_cnt", "0"))
                
            count += 1
            if count % 5000000 == 0:
                print(f"已处理 {count} 条记录...")
                
    # 3. 写入 TXT 文件
    for key_type, output_path in output_files.items():
        print(f"正在写入结果到 {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            for dual_key, items in dual_index_seqs[key_type].items():
                # 写入格式为 userId_cateId: val1,val2,...
                f.write(f"{dual_key}:{','.join(items)}\n")
            
    print(f"处理完成！")
    print(f"总记录数: {count}")
    print(f"生成的双索引序列组数: {len(dual_index_seqs['item_id'])}")
    print(f"输出文件: {', '.join(output_files.values())}")

if __name__ == "__main__":
    main()
