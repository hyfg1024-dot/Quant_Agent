#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$ROOT_DIR/apps/trading"
LAUNCHER_PATH="$HOME/Desktop/启动Quant_Agent.command"

if command -v python3.11 >/dev/null 2>&1; then
  PY_BIN="python3.11"
else
  PY_BIN="python3"
fi

cat > "$LAUNCHER_PATH" <<EOF
#!/bin/zsh
set -euo pipefail

APP_DIR="$APP_DIR"
PY_BIN="$PY_BIN"
cd "\$APP_DIR"

if [ ! -d "venv" ]; then
  echo "[首次启动] 创建虚拟环境..."
  "\$PY_BIN" -m venv venv
fi

source venv/bin/activate
"\$PY_BIN" -m pip install -r requirements.txt

echo "正在启动 Quant_Agent..."
exec "\$PY_BIN" -m streamlit run app.py --server.headless false
EOF

chmod +x "$LAUNCHER_PATH"
xattr -d com.apple.quarantine "$LAUNCHER_PATH" 2>/dev/null || true

echo "已生成桌面一键启动按钮：$LAUNCHER_PATH"
echo "请双击桌面的“启动Quant_Agent.command”启动程序。"
