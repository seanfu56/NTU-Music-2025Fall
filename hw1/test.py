#!/usr/bin/env python3
import json
import argparse
from collections import defaultdict

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def has_overlap(a_list, b_list, k):
    return len(set(a_list[:k]) & set(b_list[:k])) > 0

def compare_files(file1, file2, verbose=False):
    d1 = load_json(file1)
    d2 = load_json(file2)

    keys1, keys2 = set(d1.keys()), set(d2.keys())
    common_keys = sorted(keys1 & keys2)
    only_in_1 = sorted(keys1 - keys2)
    only_in_2 = sorted(keys2 - keys1)

    cnt_top1_same = 0
    cnt_top2_overlap = 0
    cnt_top3_overlap = 0

    ids_top1_same = []
    ids_top2_overlap = []
    ids_top3_overlap = []

    for k in common_keys:
        a, b = d1[k], d2[k]
        if not (isinstance(a, list) and isinstance(b, list) and len(a) >= 1 and len(b) >= 1):
            continue  # 跳過格式不正確者

        # Top1 相同（位置與內容都要同一個）
        if a[0] == b[0]:
            cnt_top1_same += 1
            ids_top1_same.append(k)

        # 前 2 名有重疊（忽略順序）
        if len(a) >= 2 and len(b) >= 2 and has_overlap(a, b, 2):
            cnt_top2_overlap += 1
            ids_top2_overlap.append(k)

        # 前 3 名有重疊（忽略順序）
        if len(a) >= 3 and len(b) >= 3 and has_overlap(a, b, 3):
            cnt_top3_overlap += 1
            ids_top3_overlap.append(k)

    total = len(common_keys)
    def pct(x): 
        return f"{(x/total*100):.2f}%" if total else "N/A"

    print(f"比較檔案：{file1}  vs  {file2}")
    print(f"共同 ID 數：{total}")
    if only_in_1:
        print(f"只在檔案1的 ID（{len(only_in_1)}）：{', '.join(only_in_1[:10])}{' ...' if len(only_in_1)>10 else ''}")
    if only_in_2:
        print(f"只在檔案2的 ID（{len(only_in_2)}）：{', '.join(only_in_2[:10])}{' ...' if len(only_in_2)>10 else ''}")

    print("\n=== 統計（忽略順序的重疊規則） ===")
    print(f"Top1 相同                 : {cnt_top1_same} / {total} ({pct(cnt_top1_same)})")
    print(f"前2名有重疊（Top2 overlap） : {cnt_top2_overlap} / {total} ({pct(cnt_top2_overlap)})")
    print(f"前3名有重疊（Top3 overlap） : {cnt_top3_overlap} / {total} ({pct(cnt_top3_overlap)})")

    if verbose:
        def show_list(name, lst):
            print(f"\n{name}（{len(lst)}/{total}）：")
            if not lst:
                print("(無)")
            else:
                # 避免輸出過長，只示範前 50 筆
                print(", ".join(lst[:50]) + (" ..." if len(lst) > 50 else ""))

        show_list("Top1相同的ID", ids_top1_same)
        show_list("前2名有重疊的ID", ids_top2_overlap)
        show_list("前3名有重疊的ID", ids_top3_overlap)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="比較兩個 [id -> top3清單] JSON 檔的相似度（忽略順序的重疊統計）")
    parser.add_argument("--file1", help="第一個 JSON 檔路徑", default='hc.json')
    parser.add_argument("--file2", help="第二個 JSON 檔路徑", default='output_w2v2/test_predictions.json')
    parser.add_argument("--verbose", action="store_true", help="顯示各類別命中的 ID 清單（截斷顯示）")
    args = parser.parse_args()
    compare_files(args.file1, args.file2, verbose=args.verbose)