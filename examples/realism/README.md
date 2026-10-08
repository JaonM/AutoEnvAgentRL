# 采购沙箱拟真示例

使用主链路的生成脚手架和运行时，演示：

1. 其他订单占用库存，查询只能看到剩余库存。
2. 用户拒绝拆单，重新提议后才授予采购权限。
3. 订单已提交但 HTTP 响应丢失；查询状态后可用同一个幂等键取回结果。
4. 支付暂时失败，此时库存和订单已经部分成功。
5. 用户取消活动，撤销采购授权并授权取消。
6. 取消订单、退款、释放预留库存；补偿不完整时奖励为 0，完整终态为 1。

示例是确定性运行时验收资产，不代表模型生成良品率或 `training_ready`。

## 一条命令验证

```bash
PYTHONPATH=src:. .venv/bin/python examples/realism/check.py
PYTHONPATH=src:. .venv/bin/python examples/realism/check.py --docker
```

`--output` 指定示例目录，`--base-image` 指定 Docker 基础镜像。Docker 模式随机
绑定本机端口，并自动清理自己的测试容器。报告包含镜像 ID 与运行时代码哈希。

## 本地运行

在项目根目录运行：

```bash
PYTHONPATH=src:. .venv/bin/python examples/realism/procurement.py
PYTHONPATH=output/procurement-realism .venv/bin/python examples/realism/exercise.py \
  --report output/procurement-realism/realism_report.json
```

生成器 `--output` 默认 `output/procurement-realism`。它会重写该目录内的示例
契约和脚手架，勿指向需要保留的真实任务目录。业务状态保存在该沙箱的
`.runtime/episodes.sqlite3`；reset 创建独立 episode。

## Docker HTTP 验收

```bash
docker build -t autoenv-realism-procurement:local output/procurement-realism
docker run --rm -d --name autoenv-realism-demo \
  -p 127.0.0.1:18080:8080 \
  -e SANDBOX_TRAINER_API_KEY=realism-local-test \
  autoenv-realism-procurement:local
.venv/bin/python examples/realism/exercise.py --url http://127.0.0.1:18080 \
  --report output/procurement-realism/docker_report.json
docker stop autoenv-realism-demo
```

镜像默认 `python:3.12-slim`，可通过 `docker build --build-arg BASE_IMAGE=...`
指定已有 Python 镜像。示例不需要模型凭据，也不安装额外 Python 依赖。
`realism-local-test` 仅用于绑定本机回环地址的测试实例。

## 配置与训练

业务策略字段为 `business_lifecycle`；详见 `docs/code_agent_authoring.md`。
配置以 task.json 为载体进入主构建链路。旧任务缺少该字段时保持原行为。
RL 工具调用会传递原生 call ID 作为幂等键；相同业务请求的新 call ID 是新的
操作，策略在不确定结果时应查询状态，不能假定新请求会自动去重。

只有契约声明的故障会作为可恢复工具反馈进入 rollout；真实网络断开、未声明
5xx 和会话错误仍按基础设施失败处理。时钟、已触发事件、调用计数、授权和
业务状态都在 episode 数据库中，恢复必须保留数据库，而非仅复制聊天文本。

该示例的故障按固定调用次数触发。通用事件支持 `jitter_ticks`，根据 episode
seed 和事件 ID 生成并持久化触发时间。不使用真实 sleep 增加训练耗时。
