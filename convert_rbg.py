from pathlib import Path
from PIL import Image
from tqdm import tqdm

# 指定要遍历的文件夹路径
folder_path = Path("result/segres")  # 将这里替换为你的文件夹路径

png_files = list(folder_path.rglob("*.png"))

# 使用 tqdm 显示进度条
for png_file in tqdm(png_files, desc="Processing PNG Files", unit="file"):
    try:
        if png_file.name == "final":
            continue
        print(f"start process {png_file}")
        # 打开图片
        with Image.open(png_file) as img:
            # 确保图片是 RGB 模式
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # 交换 R 和 B 通道
            r, g, b = img.split()
            img = Image.merge("RGB", (b, g, r))
            
            # 保存到原文件
            img.save(png_file)
            print(f"已处理: {png_file}")
    except Exception as e:
        print(f"处理文件 {png_file} 时出错: {e}")