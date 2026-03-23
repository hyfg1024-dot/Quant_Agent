# Fundamental Module（基本面独立板块）

该模块用于独立执行“慢引擎”基本面评估，不与交易快引擎混在一起。

## 当前能力（首版）
- 股票池管理（加入/删除，区分持仓与观察）
- 基本面评分总表（代码、名称、评分、类型、股息率、打开）
- 单只股票八维评分卡
- 总结性文本（可直接阅读）
- 复制 JSON + DeepSeek 分析（分析文本在页面下方显示）

## 八维框架
1. 生意质量  
2. 盈利能力  
3. 现金流质量  
4. 资产负债安全  
5. 增长质量  
6. 管理层配置  
7. 估值安全边际  
8. 风险控制  

## 启动
```bash
cd /Users/wellthen/Desktop/TEST/Quant_System/apps/fundamental
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m streamlit run app.py
```

## 说明
- 版本号：`FND-20260323-02`
- 默认会优先使用缓存，避免频繁抓取。
- 若某些接口临时不可用，会降级并提示数据覆盖率，不会导致页面崩溃。
- 侧栏支持填写 DeepSeek 用户名与 API Key，本地持久化到 `data/local_user_prefs.json`（已忽略上传）。
