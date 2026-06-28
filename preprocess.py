import os
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import glob

src_dir = "./data/raw_chairs"      
dst_dir = "./data/chairs/images/0"  
os.makedirs(dst_dir, exist_ok=True)

patterns = [
    os.path.join(src_dir, "**/*.png"),
    os.path.join(src_dir, "**/*.jpg"),
    os.path.join(src_dir, "**/*.jpeg"),
]

files = []
for pat in patterns:
    files.extend(glob.glob(pat, recursive=True))

print(f"Found {len(files)} images")

success_count = 0
fail_count = 0
fail_log = []  

for src_path in tqdm(files):
    try:
        fname = os.path.basename(src_path)
        dst_path = os.path.join(dst_dir, fname)
        
        counter = 1
        while os.path.exists(dst_path):
            name, ext = os.path.splitext(fname)
            dst_path = os.path.join(dst_dir, f"{name}_{counter}{ext}")
            counter += 1

        img = Image.open(src_path).convert('L')      
        img = img.resize((64, 64), Image.BICUBIC)   
        img.save(dst_path)
        success_count += 1
        
    except UnidentifiedImageError:
        fail_count += 1
        fail_log.append(f"UnidentifiedImageError: {src_path}")
        continue
        
    except PermissionError:
        fail_count += 1
        fail_log.append(f"PermissionError: {src_path}")
        continue
        
    except OSError as e:
        fail_count += 1
        fail_log.append(f"OSError: {src_path} - {str(e)}")
        continue
        
    except Exception as e:
        fail_count += 1
        fail_log.append(f"UnknownError: {src_path} - {str(e)}")
        continue

print(f"\n完成！")
print(f"成功: {success_count} 张图片")
print(f"失败: {fail_count} 张图片")
print(f"输出目录: {dst_dir}")

