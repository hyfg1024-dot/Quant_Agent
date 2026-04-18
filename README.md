# Quant_Agent

`Quant_Agent` 是一个本地化的 Streamlit 量化研究工作台，覆盖从选股、研究、交易观察到回测与模拟实盘的完整流程。

## 当前模块

| 模块 | 说明 | 入口 |
| --- | --- | --- |
| 大过滤器 | 全市场更新、条件筛选、模板化执行、导出 | `apps/filter/app.py` |
| 基本面分析 | 八维评分、新闻催化、研报摘要、AI解读 | `apps/fundamental/app.py` |
| 交易面分析 | 股票池、盘口/分时、交互K线、多智能体分析 | `apps/trading/app.py` |
| 回测系统 | YAML策略、Universe管理、回测报告输出 | `apps/backtest/run_backtest.py` |
| 模拟实盘 | 逐日推进、状态快照、策略看板 | `apps/backtest/paper_trade.py` |
| 仓位风控 | 持仓录入、PnL监控、ATR仓位建议 | `apps/portfolio/app.py` |

## 运行环境（基线）

- macOS
- Python **3.11**（推荐固定）
- `pip` 可用
- 首次安装依赖时可访问 PyPI

> 说明：当前如果使用 Python 3.9，部分依赖（如 urllib3/OpenSSL 组合）会出现告警，且新语法/新组件兼容性更弱。

## 快速启动

### 1) 交易主入口（集成导航）

```bash
cd apps/trading
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1
```

访问：[http://127.0.0.1:8501](http://127.0.0.1:8501)

### 2) 回测（CLI）

```bash
cd apps/backtest
python3.11 -m pip install -r requirements.txt
python3 run_backtest.py --config config/strategies/realestate_example.yaml
```

### 3) 模拟实盘（CLI）

```bash
cd apps/backtest
python3 paper_trade.py dashboard
```

### 4) 预警守护进程（可选）

```bash
cd /path/to/Quant_Agent
python3 daemon/alert_worker.py --once --force
python3 daemon/alert_worker.py
```

规则文件：`config/alert_rules.yaml`

## 目录结构

```text
Quant_Agent/
├── apps/
│   ├── trading/
│   ├── fundamental/
│   ├── filter/
│   ├── backtest/
│   └── portfolio/
├── shared/
├── config/
├── daemon/
├── data/                # 仅本地运行态，不建议纳入版本管理
└── docs/
```

## 运行态数据策略（重要）

本仓库遵循“源码入库，运行态数据本地化”的原则。以下内容默认只保存在本机：

- `data/backups/`
- `data/quant_system.duckdb`
- `apps/backtest/paper_trades/`
- `daemon/.alert_state.json`
- 报告和缓存目录（`apps/backtest/reports/` 等）

如果你需要跨机器迁移运行态数据，请通过压缩包或对象存储同步，不建议直接提交到 Git。

## 开发建议

- 优先保持 `apps/*` 业务逻辑与 `shared/*` 基础能力解耦。
- 新功能先在对应模块内闭环，再抽象到 `shared/`。
- 提交前至少执行：

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m compileall -q apps shared daemon
```

