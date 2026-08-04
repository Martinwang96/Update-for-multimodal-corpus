# Design

## Theme

浅色主题。暖白背景（避免纯白刺眼），白色内容卡片，深墨文字。场景：研究者白天在明亮办公室长时间阅读数据。

## Color Palette

OKLCH 定义（附 hex 参考），中性色全部向暖琥珀方向微 tint，杜绝纯黑纯白：

| Token | OKLCH | Hex 参考 | 用途 |
|---|---|---|---|
| bg | oklch(0.978 0.004 85) | #F7F5F1 | 页面背景 |
| surface | oklch(0.995 0.002 85) | #FEFDFB | 卡片/面板 |
| border | oklch(0.91 0.005 80) | #E6E3DC | 默认边框 |
| border-strong | oklch(0.85 0.008 75) | #D6D2C8 | 悬停/强调边框 |
| text | oklch(0.24 0.008 60) | #292524 | 主文字 |
| text-secondary | oklch(0.45 0.01 65) | #5C574F | 次要文字 |
| text-muted | oklch(0.63 0.01 70) | #9C9689 | 辅助/hint |
| accent | oklch(0.55 0.11 60) | #8C5A17 | 主按钮、链接、选中态 |
| accent-hover | oklch(0.48 0.10 58) | #75490F | 按钮悬停 |
| accent-soft | oklch(0.95 0.02 75) | #F5EDE0 | 选中底色、徽标 |
| success | oklch(0.55 0.11 155) | #15803D | 完成 |
| warning | oklch(0.62 0.13 85) | #B45309 | 排队 |
| error | oklch(0.52 0.14 25) | #B3261E | 错误 |
| info | oklch(0.50 0.07 240) | #3D5A99 | 进行中 |

色彩策略：Restrained。中性暖灰为主，深琥珀 accent 仅用于主操作与选中态（<10% 面积）。

## Typography

单一 sans 族：`-apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif`。数据值使用 `font-variant-numeric: tabular-nums`。

| 层级 | size/weight |
|---|---|
| 页面标题 | 20px / 600 |
| 面板标题 | 15px / 600 |
| 正文 | 14px / 400 |
| 表单标签 | 13px / 500 |
| 辅助文字 | 12px / 400 |
| 按钮 | 13px / 500 |

## Components

- 按钮：纯色填充（accent 白字主按钮；白底灰边次按钮），6px 圆角，无渐变。状态：default/hover/active/disabled 齐全
- 输入框/下拉：白底、border 1px、6px 圆角，focus 时边框 accent + 2px 透明光环
- 卡片：surface 底、border 1px、10px 圆角、极轻阴影（0 1px 2px rgba(0,0,0,.04)）
- 模块切换：segmented tabs（浅灰容器 + 白色选中块），非卡片网格
- 上传区：浅底色 + 细虚线边框，hover 边框转 accent
- 状态条：pending=warning 底、running=info 底、done=success 底、error=error 底，均为 8% 透明度浅色块 + 深色文字

## Layout

- 顶部白色细边 nav：左产品名，右服务状态指示（OpenFace / MMPose 在线点）
- 内容 max-width 1040px 居中，面板纵向堆叠
- 面板内步骤流：步骤指示器（1 上传 → 2 预览参数 → 3 结果）常驻面板顶部
- 间距节奏：面板 padding 28px，区块间隔 20px，表单组 16px

## Motion

仅状态反馈：tab 切换 150ms、上传区 hover 150ms、结果区淡入 200ms。ease-out。无入场编排动画。
