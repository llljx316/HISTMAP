# 检查是否传入参数
if [ $# -eq 0 ]; then
    echo "No arguments provided. Please provide a filename."
    exit 1
fi

# 获取第一个参数作为文件名
FILENAME=$1

while true; do
    # 检查程序是否正在运行
        if ! pgrep -f "python maskGenerator.py" > /dev/null; then
		        echo "Program stopped. Restarting $FILENAME..."
			        python $FILENAME&
				    fi
				        sleep 1000
				done
