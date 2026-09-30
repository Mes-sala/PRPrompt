# 五组优先主线服务器入口

使用现有解释器 `/root/autodl-tmp/prog_env/bin/python`。每个主线目录均包含方案说明、相关工作、代码、证据、复现计划。

先查看复现命令：

```bash
/root/autodl-tmp/prog_env/bin/python <主线目录>/reproduce.py --out <新的输出目录>
```

需要实际训练时增加 `--execute`，禁止覆盖历史输出。共享代码和数据在 `/root/ProG`，Actor资产在 `/root/autodl-tmp/accuracy_prompt_20260923/assets5090`。

原本地包中的 `完整实验记录` 链接不上传，以避免复制全部历史。完整历史及被清理文件保存在本地 `/Users/shq/Research/GP/持续推进/`，远端清理凭证在 `/root/cleanup_five_20260924/`。

本次仅检查文件SHA256和复现入口，没有重新训练。五组之外的方案不自动继续，包括R172待确认方案。
