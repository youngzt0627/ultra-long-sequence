import csv
import os
from collections import defaultdict

def main():
    data_dir = "data"
    input_path = os.path.join(data_dir, "train_joined_logs.csv")
    
    # 定义输出文件路径，带有 short 前缀
    output_files = {
        "item_id": os.path.join(data_dir, "shortUserId2itemId.txt"),
        "cate_id": os.path.join(data_dir, "shortUserId2cateId.txt"),
        "brand": os.path.join(data_dir, "shortUserId2brand.txt"),
        "u_cate_click_cnt": os.path.join(data_dir, "shortUserId2uCateClickCnt.txt"),
        "u_brand_click_cnt": os.path.join(data_dir, "shortUserId2uBrandClickCnt.txt")
    }
    
    # 使用 defaultdict(list) 存储序列
    user_seqs = {
        "item_id": defaultdict(list),
        "cate_id": defaultdict(list),
        "brand": defaultdict(list),
        "u_cate_click_cnt": defaultdict(list),
        "u_brand_click_cnt": defaultdict(list)
    }
    
    # 1. 遍历 train_joined_logs.csv 提取点击序列 (clk=1) 并截断为 20
    print("正在处理 train_joined_logs.csv 并提取短序列（包含交叉特征, max=20）...")
    if not os.path.exists(input_path):
        print(f"错误: 找不到输入文件 {input_path}，请先运行 build_joined_logs.py。")
        return

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            if row.get("clk") == "1":
                uid = row["user_id"]
                # 仅在长度未满 20 时添加
                if len(user_seqs["item_id"][uid]) < 20:
                    user_seqs["item_id"][uid].append(row["item_id"])
                    user_seqs["cate_id"][uid].append(row["cate_id"])
                    user_seqs["brand"][uid].append(row["brand"] or "NULL")
                    # 提取交叉特征短序列
                    user_seqs["u_cate_click_cnt"][uid].append(row.get("u_cate_click_cnt", "0"))
                    user_seqs["u_brand_click_cnt"][uid].append(row.get("u_brand_click_cnt", "0"))
            
            count += 1
            if count % 5000000 == 0:
                print(f"已处理 {count} 条日志...")
    
    # 2. 写入 TXT 文件
    for key, output_path in output_files.items():
        print(f"正在写入 {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            for user_id, items in user_seqs[key].items():
                f.write(f"{user_id}:{','.join(items)}\n")
            
    print(f"短序列处理完成！\n总用户数: {len(user_seqs['item_id'])}\n输出文件: {', '.join(output_files.values())}")

if __name__ == "__main__":
    main()
