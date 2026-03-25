# Quant_System

Quant 多板块总仓。当前对外主入口是 `apps/trading/app.py`（内置“基本面 / 交易面”切换）。

## 3分钟傻瓜安装（macOS）

1. 下载项目
- 方式A：`git clone https://github.com/hyfg1024-dot/Quant_System.git`
- 方式B：在 GitHub 页面点 `Code -> Download ZIP`

2. 打开项目根目录，双击运行：
- `create_desktop_launcher.command`

3. 桌面会自动生成启动按钮：
- `启动Quant_System.command`

4. 双击桌面按钮即可启动
- 首次启动会自动创建虚拟环境并安装依赖（会慢一点）
- 之后启动会快很多

5. 浏览器打开地址
- `http://localhost:8501`

## 手动启动（备用方案）

```bash
cd apps/trading
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 一键桌面启动说明

- 生成器脚本：`create_desktop_launcher.command`
- 桌面启动按钮：`~/Desktop/启动Quant_System.command`
- 启动按钮逻辑：
  - 自动进入 `apps/trading`
  - 自动检测并创建 `venv`
  - 自动安装/更新依赖
  - 启动 Streamlit

## 当前模块

- `apps/trading/`：交易指标分析 + DeepSeek 分析（页内展示）
- `apps/fundamental/`：基本面独立模块（可单独运行）
- `apps/filter/`：预留过滤器模块
- `shared/`：共享组件
- `docs/`：文档

## 版本信息（Trading）

- 代码内版本号：`QDB-20260323-DSWIN-03`
- 最新改动：交易面 DeepSeek 结果与基本面一致，采用 Markdown 排版 + 可复制文本框

## 隐私与本地数据

- DeepSeek API Key 仅保存本地：`data/local_user_prefs.json`
- 缓存/分析临时文件均已在 `.gitignore` 忽略，不会上传
