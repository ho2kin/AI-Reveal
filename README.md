# AI-Reveal

本地 LaTeX / PDF 论文 AI 率检测工具。

从 LaTeX 项目（`.tex`）或 PDF 论文中提取有效正文内容，调用腾讯 EdgeOne Makers 内置的
**朱雀检测模型**（`@makers/zhuque-text`）检测 AI 生成率，并用 **jev 决策模型**
（`@makers/jev`）对疑似 AI 段落做二次复核、给出修改建议，最终生成逐段标注的 HTML 报告。

## 功能

- **LaTeX 提取**：自动定位主文件、递归跟进 `\input`/`\include`，剥离序言、注释、
  公式、引用命令、浮动体等无关语法，只保留正文散文
- **PDF 提取**：基于 PyMuPDF，去除页眉页脚、默认截断参考文献部分
- **AI 率检测**：朱雀模型逐段标注（人工 / 疑似 AI / AI），按长度加权汇总整体占比
- **jev 复核**：对疑似 AI 段落做结构化二次判定（降低误报），并为确认可疑的段落
  生成修改建议；jev 调用失败时自动降级为仅朱雀结果
- **报告**：`report.json`（机器可读）+ `report.html`（自包含单文件，离线可看，
  总览仪表 + 逐段红黄绿标注 + 复核结论与建议）

## 安装

```bash
python -m venv .venv
.venv/Scripts/pip install -e .        # Windows
# .venv/bin/pip install -e .          # Linux/macOS
```

## 配置 API Key

三选一（优先级从高到低）：

1. 命令行 `--api-key sk-xxx`
2. 环境变量 `MAKERS_MODELS_KEY`
3. 项目根目录 `.env` 文件：`MAKERS_MODELS_KEY=sk-xxx`

Key 在 EdgeOne 控制台 Makers → Models → API Key 页面创建。朱雀模型每月免费 50 万 token，
实际扣减额度以响应中 `makers_models_usage.total_tokens` 为准。

## 使用

```bash
ai-reveal detect thesis.tex                    # 单个 .tex 文件
ai-reveal detect ./my-paper/                   # LaTeX 项目目录（自动找主文件）
ai-reveal detect paper.pdf                     # PDF 论文
ai-reveal detect paper.tex --dry-run           # 只做提取+分段，不调 API，核对提取质量
ai-reveal detect paper.tex --no-jev            # 跳过 jev 复核，仅朱雀检测
ai-reveal detect paper.tex --include-captions --keep-references
ai-reveal detect paper.tex -o my-report        # 指定报告输出目录
```

常用选项：

| 选项 | 说明 |
|---|---|
| `--api-key KEY` | 覆盖环境变量 / .env 中的 Key |
| `--chunk-chars N` | 单次请求文本块大小，默认 4000 字符 |
| `--include-captions` | 保留图表标题（默认丢弃浮动体） |
| `--keep-references` | 保留参考文献部分（默认去除） |
| `--no-jev` | 跳过 jev 复核与建议 |
| `--dry-run` | 只提取与分段，不消耗 token |
| `-o DIR` | 报告输出目录，默认 `./ai-reveal-report` |

输出：终端打印总 AI 率 / 疑似率、token 消耗、本月剩余额度、Top 可疑段落摘要与报告路径。

## 额度显示

网关未提供额度查询接口，工具按 API Key 在本地累计每月的 Makers 计费 token
（账本存于 `~/.ai-reveal/usage.json`，月底或换 Key 自动重置），结合每月免费
额度（默认 50 万 token）计算已用/剩余/百分比。如额度不同可通过环境变量覆盖：

```bash
set AI_REVEAL_MONTHLY_TOKENS=1000000   # PowerShell 示例
```

## 局限与说明

- 朱雀模型仅支持文本检测；检测结果为模型置信度参考，不构成学术不端的最终认定
- LaTeX 数学公式、表格数据等非散文内容不参与检测（可用 `--include-captions` 保留图注）
- PDF 提取质量依赖排版，双栏论文已做列序处理，但复杂版式仍可能有残缺

## 开发

```bash
.venv/Scripts/python -m pytest tests/ -v    # Windows
```
