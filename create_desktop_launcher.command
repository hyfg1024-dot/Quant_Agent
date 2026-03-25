#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$ROOT_DIR/apps/trading"
LAUNCHER_PATH="$HOME/Desktop/启动Quant_System.command"

cat > "$LAUNCHER_PATH" <<EOF
#!/bin/zsh
set -euo pipefail

APP_DIR="$APP_DIR"
cd "\$APP_DIR"

if [ ! -d "venv" ]; then
  echo "[首次启动] 创建虚拟环境..."
  python3 -m venv venv
fi

source venv/bin/activate
python3 -m pip install -r requirements.txt

echo "正在启动 Quant_System..."
exec python3 -m streamlit run app.py --server.headless false
EOF

chmod +x "$LAUNCHER_PATH"

echo "已生成桌面一键启动按钮：$LAUNCHER_PATH"
echo "请双击桌面的“启动Quant_System.command”启动程序。"
