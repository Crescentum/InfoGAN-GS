import os
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import glob

src_dir = "./data/raw_chairs"      # 你解压后的根目录（包含很多 hash 子文件夹）
dst_dir = "./data/chairs/images/0"  # 扁平化输出目录
os.makedirs(dst_dir, exist_ok=True)

# 递归查找所有 png/jpg（包括子文件夹）
patterns = [
    os.path.join(src_dir, "**/*.png"),
    os.path.join(src_dir, "**/*.jpg"),
    os.path.join(src_dir, "**/*.jpeg"),
]

files = []
for pat in patterns:
    files.extend(glob.glob(pat, recursive=True))

print(f"Found {len(files)} images")

# 统计成功和失败数量
success_count = 0
fail_count = 0
fail_log = []  # 记录失败的文件

for src_path in tqdm(files):
    try:
        # 保持原文件名，如果重名则加前缀
        fname = os.path.basename(src_path)
        dst_path = os.path.join(dst_dir, fname)
        
        # 处理重名（不同子文件夹可能有同名文件）
        counter = 1
        while os.path.exists(dst_path):
            name, ext = os.path.splitext(fname)
            dst_path = os.path.join(dst_dir, f"{name}_{counter}{ext}")
            counter += 1
        
        # 尝试打开并处理图像
        img = Image.open(src_path).convert('L')       # 转灰度
        img = img.resize((64, 64), Image.BICUBIC)   # 缩放到 64×64
        img.save(dst_path)
        success_count += 1
        
    except UnidentifiedImageError:
        # 无法识别的图像格式
        fail_count += 1
        fail_log.append(f"UnidentifiedImageError: {src_path}")
        continue
        
    except PermissionError:
        # 权限错误
        fail_count += 1
        fail_log.append(f"PermissionError: {src_path}")
        continue
        
    except OSError as e:
        # 其他图像相关错误（损坏文件等）
        fail_count += 1
        fail_log.append(f"OSError: {src_path} - {str(e)}")
        continue
        
    except Exception as e:
        # 其他未知错误
        fail_count += 1
        fail_log.append(f"UnknownError: {src_path} - {str(e)}")
        continue

# 打印统计信息
print(f"\n完成！")
print(f"成功: {success_count} 张图片")
print(f"失败: {fail_count} 张图片")
print(f"输出目录: {dst_dir}")

