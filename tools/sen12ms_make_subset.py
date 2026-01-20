import os
import shutil
import random
from tqdm import tqdm
from pathlib import Path

# ---------------------------------------------------------
# [설정]
# ---------------------------------------------------------
SOURCE_ROOT = "/mnt/e/sen12ms"  
DEST_ROOT = "./sen12ms"
EXTENSIONS = ('.tif', '.png', '.jpg', '.jpeg')
SAMPLES_PER_SEASON = 10000

SEED = 42
# ---------------------------------------------------------

def scan_and_pair_files(root_dir):
    paired_data = {}
    print(f"Scanning files in {root_dir}...")
    
    file_count = 0
    
    for root, dirs, files in os.walk(root_dir):
        # 1. 계절 정보 추출 (폴더명에서)
        # 예: ROIs1158_spring -> spring
        season = "unknown"
        path_parts = root.split(os.sep)
        for part in path_parts:
            if part.startswith("ROIs") and "_" in part:
                try:
                    season = part.split("_")[-1].lower()
                except:
                    pass
                break
        
        for file in files:
            if file.lower().endswith(EXTENSIONS):
                file_count += 1
                full_path = os.path.join(root, file)
                filename = os.path.splitext(file)[0]
                
                # 2. 파일명 파싱 (수정된 로직)
                # 예: ROIs1158_spring_lc_1_p235
                # 중간에 _lc_, _s1_, _s2_ 가 들어있는지 확인
                
                dtype = None
                unique_key_part = None
                
                if "_lc_" in filename:
                    dtype = "lc"
                    # 식별자 생성: _lc_ 를 제거하거나 공통된 포맷으로 변경
                    # 예: ROIs1158_spring_lc_1_p235 -> ROIs1158_spring_1_p235
                    unique_key_part = filename.replace("_lc_", "_XX_")
                elif "_s1_" in filename:
                    dtype = "s1"
                    unique_key_part = filename.replace("_s1_", "_XX_")
                elif "_s2_" in filename:
                    dtype = "s2"
                    unique_key_part = filename.replace("_s2_", "_XX_")
                
                if dtype and unique_key_part:
                    # 키가 겹칠 수 있으니 season 정보도 키에 포함하는 것이 안전
                    # (폴더 구조가 다르면 같은 파일명이 있을 수도 있으므로)
                    unique_key = unique_key_part 
                    
                    if unique_key not in paired_data:
                        paired_data[unique_key] = {"season": season}
                    
                    paired_data[unique_key][dtype] = full_path

    print(f"Total files scanned: {file_count}")
    
    # 완전한 쌍(Triplet)만 필터링
    valid_pairs = {}
    incomplete_count = 0
    for key, data in paired_data.items():
        if "lc" in data and "s1" in data and "s2" in data:
            valid_pairs[key] = data
        else:
            incomplete_count += 1

    print(f"Found {len(valid_pairs)} complete triplets. (Incomplete: {incomplete_count})")
    return valid_pairs

def copy_files(pairs, dest_root):
    if not os.path.exists(dest_root):
        os.makedirs(dest_root)
        
    print(f"Copying files to {dest_root}...")
    
    for key, data in tqdm(pairs.items()):
        season = data['season']
        
        for dtype in ['lc', 's1', 's2']:
            src_path = data[dtype]
            filename = os.path.basename(src_path)
            
            target_dir = os.path.join(dest_root, season, dtype)
            os.makedirs(target_dir, exist_ok=True)
            
            target_path = os.path.join(target_dir, filename)
            
            if not os.path.exists(target_path):
                shutil.copy2(src_path, target_path)

if __name__ == "__main__":
    random.seed(SEED)
    # print(f"Random Seed applied: {SEED}")

    all_pairs = scan_and_pair_files(SOURCE_ROOT)
    
    if len(all_pairs) > 0:
        season_groups = {}
        for key, data in all_pairs.items():
            season = data['season']
            if season not in season_groups:
                season_groups[season] = []
            season_groups[season].append(key)
        
        final_keys = []
        print(f"\nSampling up to {SAMPLES_PER_SEASON} pairs per season...")
        
        for season, keys in season_groups.items():
            count = len(keys)
            if count > SAMPLES_PER_SEASON:
                selected_keys = random.sample(keys, SAMPLES_PER_SEASON)
                print(f"  [{season}] {count} found -> {SAMPLES_PER_SEASON} sampled.")
            else:
                selected_keys = keys
                print(f"  [{season}] {count} found -> all included.")
            final_keys.extend(selected_keys)
            
        sampled_pairs = {k: all_pairs[k] for k in final_keys}
        copy_files(sampled_pairs, DEST_ROOT)
        print("Done!")
    else:
        print("No complete pairs found. Please check SOURCE_ROOT and filename patterns.")
