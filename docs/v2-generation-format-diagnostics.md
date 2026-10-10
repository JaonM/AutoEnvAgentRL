# V2 契约生成格式诊断

默认流水线继续使用已有 `source_json` 格式。`NativeCodeAgent.generate(...,
source_object=True)` 是显式诊断入口，尚未在测量流水线中启用。

诊断格式返回 `{"source": <契约对象>}`。传输形状校验与业务 schema 校验分离；
它检查单一对象封装、有限 JSON 和大小限制，不能替代 `compile_contract`、
参考轨迹、独立业务审查、奖励反例或实际 policy rollout。
输出 provenance 明确记录 `strict_contract_grammar_enforced: false`。

2026-10-11 的真实 GPT-6.1-Sol 设备替换诊断否定了“仅直接返回对象即可消除
格式问题”的假设：模型 turn 正常结束，但原始回答在字符 2321 处仍有 JSON
语法错误，桥接返回 502；诊断耗时约 145 秒。现有桥接对开放对象未启用严格
输出 schema，不能声称该格式具有严格解码保证。失败记录保存在
`output/env_factory_v2_iteration/native_source_object_replacement_01/`。

下一步验证闭合、递归的结构化传输表示：覆盖动态键、数组、所有 JSON 标量和
完整表达式结构；本地解码拒绝重复键及非法结构；业务编译器保持独立。
启用前必须通过真实复杂任务、业务反例与可比调用/耗时验证，不能用格式通过
代替沙箱良品或以更简单任务代替复杂任务。
