# 词项 Prompt 有限超参扩展

原 EBFP 不修改；本目录按用户授权探索五点网格和四种checkpoint。每折重建词项统计，完整与简化/随机控制保持相同搜索预算。`selected_*` 根据support CV选出，query仅评分。

`docs/01_protocol.md` 为首轮协议，`docs/02_support_stability.md` 为独立一批support的稳定性检查。`results/stage_a` 与 `results/stability` 分开报告。`provenance` 保存服务器checkpoint及已有元数据，元数据标注ProG-adapted，不声明官方忠实复现。

服务器使用已有 `/root/autodl-tmp/prog_env/bin/python`，命令：

```sh
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/prog_env/bin/python BayesFeatureSweep/run_screen.py --output BayesFeatureSweep/results/stage_a
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/prog_env/bin/python BayesFeatureSweep/run_screen.py --seed-start 16092501 --seed-count 12 --output BayesFeatureSweep/results/stability
/root/autodl-tmp/prog_env/bin/python BayesFeatureSweep/verify.py BayesFeatureSweep/results/stability
python3 BayesFeatureSweep/analyze.py BayesFeatureSweep/results/stability
```

输出目录必须不存在。验证脚本检查预测重算、源码和checkpoint哈希、support/query隔离、CV选择和冻结；不能替代效果判断或证明跨图泛化。
