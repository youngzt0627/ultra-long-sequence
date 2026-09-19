import argparse
import csv
import os
import datetime
from collections import defaultdict

def load_csv_to_dict(path, key_col):
    """通用函数：加载 CSV 到字典，以指定列为 Key。"""
    if not os.path.exists(path):
        print(f"警告: 文件 {path} 不存在。")
        return {}, []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = {row[key_col]: row for row in reader if key_col in row}
        return data, reader.fieldnames

def main():
    parser = argparse.ArgumentParser(description="将点击日志全量关联用户与广告特征。")
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录")
    parser.add_argument("--input_file", type=str, default="raw_sample.csv", help="输入的点击日志文件名")
    args = parser.parse_args()

    # 1. 加载特征表到内存
    user_map, user_cols = load_csv_to_dict(os.path.join(args.data_dir, "user_profile.csv"), "userid")
    ad_map, ad_cols = load_csv_to_dict(os.path.join(args.data_dir, "ad_feature.csv"), "adgroup_id")

    input_path = os.path.join(args.data_dir, args.input_file)
    test_start_ts = 1494604800  # 5.13 00:00:00

    # 1.5 第一遍遍历：统计用户偏好 (仅基于训练集时间范围)
    # 我们统计用户在每个类目和品牌下的总点击数
    print("第一阶段：正在统计用户偏好（基于训练集数据）...")
    user_cate_clicks = defaultdict(int)
    user_brand_clicks = defaultdict(int)
    
    with open(input_path, newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        counts = 0
        for row in reader:
            ts = int(row.get("time_stamp", 0))
            # 仅统计训练集内的点击
            if ts < test_start_ts and row.get("clk") == "1":
                uid = row.get("user")
                iid = row.get("adgroup_id")
                a_feat = ad_map.get(iid)
                if a_feat:
                    cate_id = a_feat.get("cate_id", "")
                    brand_id = a_feat.get("brand", "")
                    if cate_id:
                        user_cate_clicks[f"{uid}_{cate_id}"] += 1
                    if brand_id:
                        user_brand_clicks[f"{uid}_{brand_id}"] += 1
            
            counts += 1
            if counts % 5000000 == 0:
                print(f"已扫描 {counts} 条原始日志用于统计...")

    # 2. 第二阶段：全量关联与划分
    train_out_path = os.path.join(args.data_dir, "train_joined_logs.csv")
    test_out_path = os.path.join(args.data_dir, "test_joined_logs.csv")
    
    with open(input_path, newline="", encoding="utf-8") as f_in, \
         open(train_out_path, "w", newline="", encoding="utf-8") as f_train, \
         open(test_out_path, "w", newline="", encoding="utf-8") as f_test:
        
        reader = csv.DictReader(f_in)
        user_extras = [c for c in user_cols if c != "userid"]
        ad_extras = [c for c in ad_cols if c != "adgroup_id"]
        
        # 交叉特征列名
        cross_cols = ["u_cate_click_cnt", "u_brand_click_cnt"]
        base_cols = ["user_id", "item_id", "clk", "time_stamp"]
        fieldnames = base_cols + user_extras + ad_extras + cross_cols
        
        train_writer = csv.DictWriter(f_train, fieldnames=fieldnames, extrasaction='ignore')
        test_writer = csv.DictWriter(f_test, fieldnames=fieldnames, extrasaction='ignore')
        
        train_writer.writeheader()
        test_writer.writeheader()

        counts = {"total": 0, "train": 0, "test": 0}
        print("第二阶段：正在执行 Join 并应用交叉特征...")
        for row in reader:
            counts["total"] += 1
            uid = row.get("user")
            iid = row.get("adgroup_id")
            ts = int(row.get("time_stamp", 0))
            
            u_feat = user_map.get(uid)
            a_feat = ad_map.get(iid)
            
            if u_feat and a_feat:
                cate_id = a_feat.get("cate_id", "")
                brand_id = a_feat.get("brand", "")
                
                # 构造输出行，包含交叉特征
                out_row = {
                    "user_id": uid,
                    "item_id": iid,
                    "clk": row.get("clk"),
                    "time_stamp": ts,
                    "u_cate_click_cnt": user_cate_clicks.get(f"{uid}_{cate_id}", 0),
                    "u_brand_click_cnt": user_brand_clicks.get(f"{uid}_{brand_id}", 0)
                }
                out_row.update({k: u_feat.get(k, "") for k in user_extras})
                out_row.update({k: a_feat.get(k, "") for k in ad_extras})
                
                if ts >= test_start_ts:
                    test_writer.writerow(out_row)
                    counts["test"] += 1
                else:
                    train_writer.writerow(out_row)
                    counts["train"] += 1
            
            if counts["total"] % 5000000 == 0:
                print(f"已处理 {counts['total']} 条记录...")

    print(f"处理完成！")
    print(f"总日志数: {counts['total']}")
    print(f"训练集关联数: {counts['train']} (输出: {train_out_path})")
    print(f"测试集关联数: {counts['test']} (输出: {test_out_path})")

if __name__ == "__main__":
    main()
