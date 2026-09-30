# R130入口目录修复

初始入口沿用smoke129字符串，source/config保护在写入前拒绝恢复，PID253513终止，没有R130预测。保留原入口/manifest/日志；另封v2入口只将smoke129改为smoke130，driver/方法矩阵/审计不变。启动前校验原smoke129 config与本地副本SHA一致，确认未改写旧批。
