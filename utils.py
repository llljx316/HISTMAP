from pathlib import Path

def iterative_all_files(directory, process_file, suffix_filter = None):
    # directory = 'dataset'  # dataset folder
    path = Path(directory)
    for file_path in path.rglob('*'):
        if file_path.is_file() and (suffix_filter is None or file_path.suffix in suffix_filter):
            process_file(file_path)

# 给定路径
file_path = Path('dataset/Bodleian Library/2229bb6b-5fad-4e2b-81ce-ccc204773598.jpg')

# 提取目录名和文件名（不包括扩展名）
dir_name = file_path.parent.name  # 提取 'Bodleian Library'
file_name = file_path.stem  # 提取 '2229bb6b-5fad-4e2b-81ce-ccc204773598'

# 组合结果
result = f"{dir_name}/{file_name}"