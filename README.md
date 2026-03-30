# Quant_System

`Quant_System` 是一个基于 Streamlit 构建的本地量化研究工作台，面向日常股票研究、观察池管理、基本面分析与条件筛选场景。项目目前以 macOS 本地运行方式为主，强调可落地、可视化和低门槛部署。

## 项目概览

当前版本包含 3 个核心模块：

| 模块 | 说明 | 入口 |
| --- | --- | --- |
| Trading | 交易观察台，聚合行情快照、盘口、分时结构与 DeepSeek 深析流程 | `apps/trading/app.py` |
| Fundamental | 基本面研究台，提供八维评分、总结文本与 DeepSeek 辅助解读 | `apps/fundamental/app.py` |
| Filter | 全市场条件筛选器，支持模板化筛选、AI 辅助设定和 Excel 导出 | `apps/filter/app.py` |

## 主要能力

- 多模块本地量化研究界面
- 股票池管理，支持持仓与观察分组
- 基本面八维评分与结构化总结
- DeepSeek 分析接入，支持本地保存用户配置
- 全市场筛选、模板保存、结果分池与 Excel 导出
- macOS 一键启动脚本与桌面快捷入口

## 目录结构

```text
Quant_System/
├── apps/
│   ├── trading/          # 交易观察模块
│   ├── fundamental/      # 基本面研究模块
│   └── filter/           # 条件筛选模块
├── shared/               # 共享 UI / 通用逻辑
├── data/                 # 本地数据、缓存、用户配置
├── docs/                 # 附加文档
├── create_desktop_launcher.command
└── README.md
```

## 运行要求

- macOS
- Python 3.9+
- 终端可用 `python3` 与 `pip`
- 首次安装依赖时可正常访问 Python 包源

## 快速开始

### 方式一：使用桌面启动脚本

在项目根目录执行：

```bash
chmod +x create_desktop_launcher.command
xattr -d com.apple.quarantine create_desktop_launcher.command 2>/dev/null || true
./create_desktop_launcher.command
```

执行完成后，桌面会生成启动入口：

- `启动Quant_System.command`

双击后即可启动默认页面。

### 方式二：手动启动模块

#### Trading

```bash
cd apps/trading
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
streamlit run app.py
```

#### Fundamental

```bash
cd apps/fundamental
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
streamlit run app.py
```

#### Filter

```bash
cd apps/filter
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
streamlit run app.py
```

## 默认访问地址

启动后默认访问：

- [http://localhost:8501](http://localhost:8501)

如果端口被占用，可以改为其他端口，例如：

```bash
streamlit run app.py --server.port 8510
```

## macOS 安全提示

如果首次执行脚本时遇到“Apple 无法验证开发者”或系统拦截，通常是 macOS 的隔离属性导致。可以执行：

```bash
chmod +x create_desktop_launcher.command
xattr -d com.apple.quarantine create_desktop_launcher.command 2>/dev/null || true
./create_desktop_launcher.command
```

如果桌面生成的启动入口被拦截，可继续执行：

```bash
xattr -d com.apple.quarantine ~/Desktop/启动Quant_System.command
chmod +x ~/Desktop/启动Quant_System.command
```

## 配置与本地数据

- DeepSeek 用户名与 API Key 仅保存在本地
- 本地偏好文件默认位于 `data/local_user_prefs.json`
- 市场快照、筛选模板和缓存均保存在项目本地目录，不会自动上传到 GitHub

建议将以下内容视为本地运行态数据，而不是源码的一部分：

- `data/`
- 各模块下的 `venv/`
- 本地缓存、数据库、导出结果

## 常见问题

### 1. 依赖安装较慢

首次安装会下载较多 Python 包，等待时间取决于网络环境。

### 2. 页面无法访问

请确认：

- 当前模块依赖已安装完成
- Streamlit 已正常启动
- 端口未被其他程序占用

### 3. API Key 是否会进入仓库

不会。项目默认将本地用户配置与缓存文件排除在 Git 之外。

## 开发说明

- 项目当前以本地运行和单仓库维护为主
- UI 共享逻辑位于 `shared/`
- 各模块均可独立启动，也可在主工作流中联动使用
- 若修改模块逻辑，建议优先在对应 `apps/<module>/` 下完成验证

## 许可证与使用说明

本仓库当前未在 README 中单独声明开源许可证。如需对外分发、商用或二次发布，建议先补充明确的许可证文件与使用条款。
