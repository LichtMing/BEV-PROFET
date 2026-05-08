import sqlite3
import numpy as np
import cv2
import os
from tqdm import tqdm
from bev2bytes import bytes2bev

db_path = "ZS-0.mtsdb.meta"

def pre_process(image):
    return (image / 255.0 - 0.5) * 2.0

def fit_288(img_81_314):
    # img is (81, 314) -> (288, 288)
    img = img_81_314[:, 13 : 13+288]
    pad_img = np.pad(img, ((103, 104), (0, 0)), 'constant')
    return pad_img

def main():
    os.makedirs('BEVData', exist_ok=True)
    os.makedirs('BEVMaskData', exist_ok=True)
    os.makedirs('BEVLabel', exist_ok=True)
    
    print("Connecting to DB...")
    con = sqlite3.connect(db_path)
    cursor = con.cursor()
    
    # 1. Load configs for global drivable and line
    cursor.execute("SELECT drivable, line, shape FROM configs LIMIT 1")
    gdb, glb, gshape_str = cursor.fetchone()
    gshape = eval(gshape_str) 
    g_blen = (gshape[0] * gshape[1] - 1) // 8 + 1
    gdrivable, gline = bytes2bev([gdb, glb], gshape, g_blen)
    
    gdrivable = fit_288(gdrivable)
    gline = fit_288(gline)
    
    # 2. Get all vehicles mapping
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'veh_%'")
    veh_tables = [x[0] for x in cursor.fetchall()]
    
    step_to_vehs = {}
    print("Caching vehicle steps...")
    for vt in tqdm(veh_tables):
        cursor.execute(f"SELECT step FROM {vt}")
        steps = [x[0] for x in cursor.fetchall()]
        for s in steps:
            if s not in step_to_vehs:
                step_to_vehs[s] = []
            step_to_vehs[s].append(vt)
            
    cursor.execute("SELECT MIN(step), MAX(step) FROM global")
    min_step, max_step = cursor.fetchone()
    
    print(f"Steps: {min_step} to {max_step}")
    
    # 3. Main Data Generation Loop
    for index in tqdm(range(69, max_step - 10, 10)):
        if index not in step_to_vehs or index-10 not in step_to_vehs or index-20 not in step_to_vehs:
            continue
            
        v20 = set(step_to_vehs[index-20])
        v10 = set(step_to_vehs[index-10])
        v0  = set(step_to_vehs[index])
        
        fu_vehs = list(v20 & v10 & v0)[:16]
        if len(fu_vehs) == 0:
            continue
            
        npy_pre = np.zeros((7, 16, 6, 72, 72), np.float32)
        mask_pre = np.zeros((7, 16), np.uint8)
        
        for veh_idx, veh in enumerate(fu_vehs):
            for t_idx in range(7):
                t_step = index - 60 + t_idx * 5
                
                cursor.execute(f"SELECT drivable, line, vehicle FROM {veh} WHERE step == ?", (t_step,))
                row = cursor.fetchone()
                
                if row is None:
                    continue 
                    
                ldb, lgb, lvb = row
                l_blen = (64 * 64 - 1) // 8 + 1
                ldrivable, lline, lvehicle = bytes2bev([ldb, lgb, lvb], (64, 64), l_blen)
                
                ldrivable = np.pad(ldrivable, ((4,4), (4,4)), 'constant')
                lline = np.pad(lline, ((4,4), (4,4)), 'constant')
                lvehicle = np.pad(lvehicle, ((4,4), (4,4)), 'constant')
                
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(lvehicle), connectivity=8)
                ego_label_idx = labels[36, 36]
                if ego_label_idx == 0 and num_labels > 1:
                    min_dist = 9999
                    for i in range(1, num_labels):
                        cx, cy = centroids[i]
                        if cx is np.nan or cy is np.nan: continue
                        dist = (cx-36)**2 + (cy-36)**2
                        if dist < min_dist:
                            min_dist = dist
                            ego_label_idx = i
                
                ego = (labels == ego_label_idx) if ego_label_idx > 0 else np.zeros_like(lvehicle)
                others = lvehicle ^ ego
                
                rgb = np.ones((72, 72, 3), np.uint8) * 255
                rgb[ldrivable] = [220, 220, 220]
                rgb[lline] = [192, 192, 192]
                rgb[others] = [65, 105, 225]
                rgb[ego] = [65, 179, 113]
                rgb_norm = pre_process(rgb).transpose(2, 0, 1) 
                
                prob = np.zeros((3, 72, 72), np.float32)
                prob[0] = np.where(ldrivable, 0.9, 0.1)
                prob[1] = np.where(lline, 0.9, 0.1)
                prob[2] = np.where(others, 0.9, 0.1)
                
                combined = np.concatenate([rgb_norm, prob], axis=0)
                npy_pre[t_idx, veh_idx] = combined
                mask_pre[t_idx, veh_idx] = 1
                
        cursor.execute("SELECT vehicle FROM global WHERE step == ?", (index - 20,))
        gveh_20 = cursor.fetchone()[0]
        cursor.execute("SELECT vehicle FROM global WHERE step == ?", (index - 10,))
        gveh_10 = cursor.fetchone()[0]
        cursor.execute("SELECT vehicle FROM global WHERE step == ?", (index,))
        gveh_0 = cursor.fetchone()[0]
        
        gv_20 = fit_288(bytes2bev(gveh_20, gshape, g_blen))
        gv_10 = fit_288(bytes2bev(gveh_10, gshape, g_blen))
        gv_0  = fit_288(bytes2bev(gveh_0, gshape, g_blen))
        
        mask_ori_label = np.concatenate([
            np.expand_dims(gdrivable, 0),
            np.expand_dims(gline, 0),
            np.expand_dims(gv_20, 0),
            np.expand_dims(gv_10, 0),
            np.expand_dims(gv_0, 0)
        ], axis=0)
        
        np.save(f"BEVData/1_{index}.npy", npy_pre)
        np.save(f"BEVMaskData/1_{index}.npy", mask_pre)
        np.save(f"BEVLabel/1_{index}.npy", mask_ori_label)

if __name__ == '__main__':
    main()