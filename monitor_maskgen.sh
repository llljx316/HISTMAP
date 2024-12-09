while true; do
    # 检查程序是否正在运行
        if ! pgrep -f "python maskGenerator.py" > /dev/null; then
		        echo "Program stopped. Restarting maskGenerator.py..."
			        python maskGenerator.py &
				    fi
				        sleep 100
				done
