# Quant Backtest

港股多空对冲通用回测系统（可配置、多板块、可扩展）。

## 1. 功能概览

- 多板块 `universe` 管理（sector/group/stocks）
- YAML 策略配置（资金、权重、再平衡、成本、止损）
- 数据层：`yfinance` 主源 + `akshare` 兜底，自动本地缓存到 `data/`
- 回测引擎：
  - 多空组合逐日推进
  - 月/周/季/日再平衡
  - 交易成本、滑点、融券日成本
  - 单标的/组合止损
  - 长停牌冻结交易（>30天）
- 指标分析：收益、风险、风险调整、对冲相关、成本拖累
- 敏感性分析：融券费率多场景回测
- 报告输出：独立 Plotly HTML（无需启动服务）

## 2. 目录结构

```text
apps/backtest/
├── config/
│   ├── universe.yaml
│   └── strategies/
│       ├── realestate_example.yaml
│       └── tech_example.yaml
├── src/
│   ├── __init__.py
│   ├── config_loader.py
│   ├── universe_manager.py
│   ├── data_manager.py
│   ├── backtest_engine.py
│   ├── metrics.py
│   └── visualizer.py
├── data/
├── reports/
├── run_backtest.py
├── manage_universe.py
└── requirements.txt
```

## 3. 安装

```bash
cd apps/backtest
pip install -r requirements.txt
```

## 4. 标的池管理

```bash
# 列表
python manage_universe.py list
python manage_universe.py list --sector real_estate
python manage_universe.py list --sector real_estate --group soe

# 添加
python manage_universe.py add-sector tech --name "科技" --description "港股科技" --benchmark "^HSTECH"
python manage_universe.py add-group tech megacap --name "大型互联网"
python manage_universe.py add-stock tech megacap 0700.HK --name "腾讯控股" --tags "社交,游戏"

# 批量导入
python manage_universe.py import-stocks --file my_stocks.csv --sector tech --group megacap

# 删除
python manage_universe.py remove-stock 0700.HK
python manage_universe.py remove-sector tech

# 校验
python manage_universe.py validate
python manage_universe.py validate --sector real_estate --start 2021-01-01 --end today
```

`import-stocks` CSV 至少包含列：`code,name`，可选 `tags`（逗号分隔）。

## 5. 运行回测

```bash
# 使用策略配置回测并生成 HTML 报告
python run_backtest.py --config config/strategies/realestate_example.yaml

# 指定输出目录
python run_backtest.py --config config/strategies/realestate_example.yaml --output reports/

# 仅验证 universe 数据可用性
python run_backtest.py --validate-universe

# 仅更新缓存，不跑回测
python run_backtest.py --update-data-only --start 2021-01-01 --end today
```

报告文件会输出到 `reports/report_{strategy_name}_{timestamp}.html`。

## 6. 策略配置要点

参考：`config/strategies/realestate_example.yaml`

关键校验：

- `long_pct + short_pct + cash_buffer_pct == 1`
- 多头权重和、空头权重和均为 `1`
- 标的代码必须存在于 `universe.yaml`
- 回测起止日期合法

## 7. 特殊处理说明

- 停牌处理：前值填充用于估值；连续缺失超过 30 天视为不可交易，冻结交易。
- 数据缺失/覆盖不完整：在报告 `warnings` 中提示。
- 退市/拉取失败标的：单只跳过，不阻断全组合（但会给出警告）。
- 风险提示：报告默认包含后视偏差与生存者偏差提醒。

## 8. 常见问题

1. `yfinance` 或 `akshare` 拉取失败
- 请检查网络或更换时间重试。
- 可先执行 `python run_backtest.py --update-data-only` 预热缓存。

2. 报告打不开
- 使用 `--no-browser` 关闭自动打开后，手动打开 `reports/` 下 HTML。

3. 某些标的行数很少
- 可能为停牌、延迟上市、退市或数据源覆盖不足。
