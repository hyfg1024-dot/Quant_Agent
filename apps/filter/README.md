# Filter Module（大过滤器）

版本：`FLT-20260327-01`

## 功能（v0.1）
- 全市场快照更新（可设置样本上限）
- 深度补充（调用 fundamental 引擎，默认补充前 N 只）
- 支持市场范围：A股 / 港股 / A+H
- 周更任务（A+H，7天门槛，未到期自动跳过）
- 20+ 可配置筛选项（像电商筛选）
- 手动筛选 + AI辅助设定（自然语言生成条件）
- 结果分池：通过池 / 排除池 / 缺失项
- 导出 Excel

## 条件维度（可勾选启用）
- A. 财务健康与硬排除：ST、立案调查、处罚、资金占用、违规减持、审计意见、负债率、经营现金流等
- B. 估值与质量：PE/PB、股息率、ROE、毛利率、净利率、商誉占比、有息负债占比
- C. 规模与流动性：总市值、换手率、量比、成交额
- D. 行业过滤：夕阳行业关键词排除（可自定义）

## 启动
```bash
cd /Users/wellthen/Desktop/Quant/Quant_System/apps/filter
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 使用流程
1. 左侧选择市场范围（A股 / 港股 / A+H），点击“更新全市场数据”
2. 设定筛选条件（或用 AI辅助设定）
3. 点击“执行筛选”
4. 在 3 个结果 tab 查看并导出 Excel

## 推荐更新策略
- 首轮建库：`市场范围=A+H`，`股票数=0(全部)`，`深度补充=500~800`，`安全模式=开启`
- 稳态周更：点击“执行周更（A+H，7天一次）”
- 防封建议：默认不勾选“忽略缓存强制重抓”，仅月度需要全量重抓时开启

## 数据文件
- `data/filter_market.db`：市场快照数据库
- `data/cache/`：深度补充缓存
- `data/filter_templates.json`：筛选模板
- `data/manual_flags.json`：管理层风控人工标注

> 说明：`manual_flags.json` 可用于补充立案调查、处罚、质押率等公开信息标记。
