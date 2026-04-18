# P0 Engineering Baseline

本文件定义当前仓库的最低工程基线（P0）。

## 1. 代码与数据分离

- 代码文件进入 Git。
- 运行态数据、缓存、报告、数据库不进入 Git。

重点路径：

- `data/backups/`
- `data/quant_system.duckdb`
- `apps/backtest/paper_trades/`
- `daemon/.alert_state.json`
- `apps/backtest/reports/`

## 2. Python 版本统一

- 推荐并默认使用 Python `3.11.x`。
- 开发机最低要求 `3.11`。

## 3. 提交前最小检查

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m compileall -q apps shared daemon
```

## 4. 提交规范

- 功能提交不混入本地运行态产物。
- 需要保留的本地状态用导出包/备份目录传输，不用 Git 传输。
- README 与模块入口保持同步。

