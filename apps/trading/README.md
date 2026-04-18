# Trading Module（交易指标分析）

本目录是从 `Quant_Ai` 迁移的交易模块，保持原有可运行能力。

## 启动
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m streamlit run app.py
```

## 说明
- 当前保留：交易面 + 基本面看板 + DeepSeek 分析流程
- API 本地缓存文件：`data/local_user_prefs.json`（已被 `.gitignore` 忽略）
