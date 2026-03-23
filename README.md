# Quant System（多板块总仓）

这是新一代模块化仓库，用于承载多个功能板块。

## 当前结构
- `apps/trading/`：交易指标分析模块（从 `Quant-AI` 首次迁移）
- `apps/fundamental/`：基本面板块（待开发）
- `apps/filter/`：大过滤器板块（待开发）
- `shared/`：公共组件与工具（待扩展）
- `docs/`：架构文档与里程碑

## 先跑交易模块
```bash
cd apps/trading
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m streamlit run app.py
```

## 版本策略
- `Quant-AI`：交易模块专仓（`v1.x` 维护线）
- `Quant_System`：多板块平台（后续 `v2.x` 主线）
