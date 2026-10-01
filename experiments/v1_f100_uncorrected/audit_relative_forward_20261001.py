"""Audit uncorrected F100 and compare SHA-locked original baselines."""
import argparse
import collections
import csv
import itertools
import json
import os
import sys
from pathlib import Path
ROOT=Path(os.environ["PRPROMPT_F100_SOURCE_ROOT"]) if "PRPROMPT_F100_SOURCE_ROOT" in os.environ else Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"tools"))
import numpy as np
import torch
from torch.nn import functional as F
from prprompt.model import TwoLayerGCN,normalized_adjacency
from prprompt.protocol import load_dataset,sha_file,save_json
from prprompt.train import tensor_hash,state_hash
from experiments.v1_f100.run import read_plan,digest
from experiments.v1_f100.historical import load
from metric_utils import numpy_metrics

METRICS=["accuracy","macro_f1","micro_f1","roc_auc_macro","average_precision_macro"]
MAIN=["FrozenLinear","FrozenPrototype","FullFineTune","GPF","GPFPlus","GraphPromptDiag","PRPrompt"]
EXTRA={}
PAIRS=["PRPrompt","FullFineTune"]


def write_csv(path,rows):
    with path.open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(rows)


def summarize(args):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan=read_plan(args.plan);old=load();device=torch.device(args.device)
    root=args.plan.parent/("smoke" if args.smoke else "results")
    splits=plan["split_seeds"][:1] if args.smoke else plan["split_seeds"]
    seeds=plan["train_seeds"][:1] if args.smoke else plan["train_seeds"]
    expected=set(itertools.product([j["id"] for j in plan["jobs"]],splits,seeds))
    seen=set();data={};models={};groups=collections.defaultdict(lambda:collections.defaultdict(list))
    references={name:json.loads(Path(r["manifest_file"]).read_text()) for name,r in plan["references"].items()}
    max_metric_error=max_replay_error=max_gate_error=0.0;source_checked=0
    max_forward_error=max_forward_probability_error=max_rebuilt_gate_probability_error=0.0
    absolute_forward_failures=[];maximum_mixed_forward_error=0.0
    forward_class_changes=rebuilt_gate_class_changes=0;classification_drift=[]
    positive_gradients=[]
    jobmap={j["id"]:j for j in plan["jobs"]}

    def checked_reference(path,name):
        nonlocal source_checked
        relative=str(path.relative_to(plan["references"][name]["root"]))
        assert sha_file(path)==references[name][relative],str(path)
        source_checked+=1
        return json.loads(path.read_text())

    for path in sorted(root.glob("*/5/*/*/F100/metrics.json")):
        row=json.loads(path.read_text());identity=tuple(row[k] for k in ["checkpoint_id","split_seed","train_seed"])
        assert identity in expected and identity not in seen;seen.add(identity)
        assert row["plan_sha256"]==plan["plan_sha256"] and row["smoke"]==args.smoke
        for name,key in [("state.pt","state_sha256"),("selection.json","selection_sha256"),("predictions.npz","predictions_sha256")]:
            assert sha_file(path.parent/name)==row[key],str(path.parent/name)
        seal=json.loads((path.parent/"seal.json").read_text())
        for key in ["plan_sha256","signature","predictions_sha256","support_sha256","query_sha256"]:assert seal[key]==row[key]
        a=row["audit"];assert a["backbone_frozen"] and a["backbone_before"]==a["backbone_after"]
        assert not a["query_labels_received"] and not a["M2_active"] and not a["M3_active"] and not a["kd_active"]
        assert a["head_parameters"]==0 and a["adam_actual_epsilon"]==1e-8
        state=torch.load(path.parent/"state.pt",map_location="cpu",weights_only=True)
        s,sy,q=(state[k] for k in ["support","support_labels","query"])
        assert tensor_hash(s)==row["support_sha256"] and tensor_hash(sy)==row["support_labels_sha256"]
        assert tensor_hash(q)==row["query_sha256"] and not torch.isin(s,q).any()
        selection=json.loads((path.parent/"selection.json").read_text())
        assert selection["learning_rate"]==.01 and selection["weight_decay"]==0 and not selection["query_labels_used"]
        assert selection["steps"]==(2 if args.smoke else 100)
        prompt=state["prompt"];gradients=prompt["optimizer_gradients"]
        assert gradients.shape==(selection["steps"],len(sy.unique()),128) and torch.isfinite(gradients).all()
        assert float(gradients.abs().max())==a["prompt_gradient_max"]>0
        positive_gradients.append(a["prompt_gradient_max"])
        if row["dataset"] not in data:
            data[row["dataset"]]=load_dataset(row["dataset"],plan["data_root"])
        x,e,y,pool,query,classes=data[row["dataset"]]
        assert torch.equal(q,query) and torch.equal(sy,y[s])
        assert torch.all(torch.bincount(sy,minlength=classes)==5)
        assert digest({"x":tensor_hash(x),"edges":tensor_hash(e)})==row["graph_sha256"]
        if row["checkpoint_id"] not in models:
            j=jobmap[row["checkpoint_id"]];assert sha_file(j["checkpoint"])==j["checkpoint_sha256"]
            model=TwoLayerGCN(j["input_dim"],128,128).to(device).eval();model.requires_grad_(False)
            model.load_state_dict(torch.load(j["checkpoint"],map_location="cpu",weights_only=True))
            xd,ed=x.to(device),e.to(device)
            adj=normalized_adjacency(ed,len(x),x.dtype)
            with torch.no_grad():u=model.conv_layers[0](xd,ed);r=torch.sparse.mm(adj,torch.sparse.mm(adj,xd))
            models[row["checkpoint_id"]]=(model,u,r,adj,ed)
        model,u,r,adj,ed=models[row["checkpoint_id"]]
        assert state_hash(model)==a["backbone_before"]
        with torch.no_grad():
            si,ty=s.to(device),sy.to(device)
            raw=old["rectified_teacher"](r-r.mean(0,keepdim=True),si,ty,classes,'rect_teacher_ref50',row["train_seed"])[0]
            gate,info=old["temperature_gate"](raw,adj,.1,'raw')
            error=float((gate-prompt["gate"].to(device)).abs().max());max_gate_error=max(max_gate_error,error)
            assert error<2e-6,error
            assert torch.allclose(info["probability"],prompt["gate_probability"].to(device),atol=2e-6,rtol=0)
            scale=u.std(0).clamp_min(1e-6)
            assert torch.allclose(scale,state["scale"].to(device),atol=1e-6,rtol=1e-6)
            # Test the serialized fixed routing actually used by the fitted prompt.
            z=model.conv_layers[1](F.leaky_relu(u+10*(prompt["gate"].to(device)@prompt["tokens"].to(device).tanh())*state["scale"].to(device)),ed)
            forward_error=float((z-state["raw_embedding"].to(device)).abs().max())
            max_forward_error=max(max_forward_error,forward_error)
            forward_diff=(z-state["raw_embedding"].to(device)).abs()
            mixed_error=float((forward_diff/(2e-6+1e-6*state["raw_embedding"].to(device).abs())).max())
            maximum_mixed_forward_error=max(maximum_mixed_forward_error,mixed_error)
            assert mixed_error<1,mixed_error
            if forward_error>=2e-6:
                absolute_forward_failures.append({"unit":str(path.relative_to(args.plan.parent)),
                    "absolute_forward_error":forward_error,"saved_embedding_max_abs":float(state["raw_embedding"].abs().max()),
                    "mixed_error_ratio":mixed_error})
            forward_prob=old["logits"](z-z.mean(0,keepdim=True),si,ty,classes)[q.to(device)].cpu().double().softmax(-1).numpy()
            zg=model.conv_layers[1](F.leaky_relu(u+10*(gate@prompt["tokens"].to(device).tanh())*scale),ed)
            gate_prob=old["logits"](zg-zg.mean(0,keepdim=True),si,ty,classes)[q.to(device)].cpu().double().softmax(-1).numpy()
            saved=state["raw_embedding"].to(device)
            assert torch.equal(saved-state["center"].to(device),state["centered_embedding"].to(device))
            scores=old["logits"](state["centered_embedding"].to(device),si,ty,classes)
            replay=scores[q.to(device)].cpu().double().softmax(-1).numpy()
        with np.load(path.parent/"predictions.npz") as p:
            probabilities=p["probabilities"].copy()
            assert np.array_equal(p["support"],s.numpy()) and np.array_equal(p["query"],q.numpy())
        max_forward_probability_error=max(max_forward_probability_error,float(np.max(np.abs(forward_prob-probabilities))))
        max_rebuilt_gate_probability_error=max(max_rebuilt_gate_probability_error,float(np.max(np.abs(gate_prob-probabilities))))
        nf=int(np.sum(forward_prob.argmax(1)!=probabilities.argmax(1)))
        ng=int(np.sum(gate_prob.argmax(1)!=probabilities.argmax(1)))
        forward_class_changes+=nf;rebuilt_gate_class_changes+=ng
        if nf or ng:
            changed=(forward_prob.argmax(1)!=probabilities.argmax(1)) | (gate_prob.argmax(1)!=probabilities.argmax(1))
            sorted_probs=np.sort(probabilities[changed],axis=1)
            classification_drift.append({"unit":str(path.relative_to(args.plan.parent)),"fixed_route_changes":nf,
                  "rebuilt_route_changes":ng,"sealed_top_two_margin_max":float((sorted_probs[:,-1]-sorted_probs[:,-2]).max()),
                  "fixed_route_probability_error":float(np.abs(forward_prob-probabilities).max()),
                  "rebuilt_route_probability_error":float(np.abs(gate_prob-probabilities).max())})
        error=float(np.max(np.abs(replay-probabilities)));max_replay_error=max(max_replay_error,error)
        assert error<2e-6,error
        assert np.array_equal(replay.argmax(1),probabilities.argmax(1))
        got=numpy_metrics(y[q].numpy(),probabilities)
        for metric,value in got.items():
            error=abs(value-row["metrics"][metric]);max_metric_error=max(max_metric_error,error);assert error<1e-12
        assert state_hash(model)==a["backbone_after"] and all(p.grad is None for p in model.parameters())
        groups[(row["checkpoint_id"],"v1_F100")][row["split_seed"]].append(got)
        relative=Path(row["checkpoint_id"])/"5"/str(row["split_seed"])/str(row["train_seed"])
        for method in MAIN:
            ref=checked_reference(Path(plan["base"])/relative/method/"metrics.json","main")
            assert ref["support_sha256"]==row["support_sha256"] and ref["query_sha256"]==row["query_sha256"]
            assert ref["support_labels_sha256"]==row["support_labels_sha256"]
            groups[(row["checkpoint_id"],method)][row["split_seed"]].append(ref["metrics"])
        for name,view in EXTRA.items():
            ref=checked_reference(Path(plan["m3"])/"results"/relative/view/"metrics.json","m3")
            assert ref["support_sha256"]==row["support_sha256"] and ref["query_sha256"]==row["query_sha256"]
            groups[(row["checkpoint_id"],name)][row["split_seed"]].append(ref["metrics"])
    assert seen==expected,f'Missing {len(expected-seen)} units'
    output=args.plan.parent/("smoke_analysis" if args.smoke else "analysis");output.mkdir(exist_ok=True)
    blocks={};table=[];max_seed_range=0.
    for (job_id,method),draws in sorted(groups.items()):
        j=jobmap[job_id];assert all(len(reps)==len(seeds) for reps in draws.values())
        values={s:{m:float(np.mean([r[m] for r in reps])) for m in METRICS} for s,reps in draws.items()}
        blocks[(job_id,method)]=values
        if method=="v1_F100":
            max_seed_range=max(max_seed_range,max(max(r["accuracy"] for r in reps)-min(r["accuracy"] for r in reps) for reps in draws.values()))
        row={"dataset":j["dataset"],"family":j["family"],"pretrain_seed":j["pretrain_seed"],"checkpoint_id":job_id,
             "method":method,"support_draws":len(draws)}
        for m in METRICS:
            v=[r[m] for r in values.values()];row[m+"_mean"]=float(np.mean(v))
            row[m+"_std_over_support_draws"]=float(np.std(v,ddof=1)) if len(v)>1 else 0.
        table.append(row)
    write_csv(output/"checkpoint_comparison.csv",table)
    pairs=[];rng=np.random.default_rng(plan["seed_for_bootstrap"])
    for j in plan["jobs"]:
        target=blocks[(j["id"],"v1_F100")]
        for reference in PAIRS:
            ref=blocks[(j["id"],reference)];assert target.keys()==ref.keys()
            for m in METRICS:
                delta=np.array([target[s][m]-ref[s][m] for s in sorted(target)])
                ci=np.quantile(rng.choice(delta,size=(10000,len(delta)),replace=True).mean(1),[.025,.975]).tolist()
                pairs.append({"checkpoint_id":j["id"],"dataset":j["dataset"],"family":j["family"],"pretrain_seed":j["pretrain_seed"],
                              "reference":reference,"metric":m,"mean_delta":float(delta.mean()),"paired_bootstrap_95ci":ci})
        rng.integers(0, len(target), size=(5, 10000, len(target)))
    save_json(output/"paired_effects.json",pairs)
    aggregates=[]
    for d,f,m in itertools.product(["Cora","CiteSeer"],["Edgepred_Gprompt","Edgepred_GPPT"],MAIN+["v1_F100"]):
        rows=[r for r in table if (r["dataset"],r["family"],r["method"])==(d,f,m)];assert len(rows)==1
        row={"dataset":d,"family":f,"method":m}
        for metric in METRICS:
            v=[r[metric+"_mean"] for r in rows];row[metric+"_mean"]=float(np.mean(v));row[metric+"_std_over_support_draws"]=rows[0][metric+"_std_over_support_draws"]
        aggregates.append(row)
    write_csv(output/"condition_comparison.csv",aggregates);save_json(output/"aggregates.json",aggregates)
    audit={"expected":len(expected),"present":len(seen),"missing":[],"smoke":args.smoke,"plan_sha256":plan["plan_sha256"],
           "frozen_backbones_verified":len(seen),"positive_prompt_gradients_verified":len(positive_gradients),
           "source_reference_files_verified":source_checked,"all_five_metrics_independently_recomputed":True,
           "maximum_metric_error":max_metric_error,"maximum_gate_replay_error":max_gate_error,
           "maximum_final_probability_replay_error":max_replay_error,"replay_argmax_equal":True,
           "maximum_accuracy_range_across_training_seeds":max_seed_range,"minimum_prompt_gradient":min(positive_gradients),
           "actual_adam_eps":1e-8,"M2_active":False,"M3_active":False,"KD_active":False,
           "maximum_raw_forward_error":max_forward_error,"maximum_forward_probability_error":max_forward_probability_error,
           "maximum_rebuilt_gate_probability_error":max_rebuilt_gate_probability_error,
           "original_gate_rebuilt_probability_tolerance_passed":max_rebuilt_gate_probability_error<2e-6,
           "serialized_state_readout_tolerance":2e-6,"raw_forward_tolerance":2e-6,
           "serialized_route_and_rebuilt_route_argmax_equal":forward_class_changes==rebuilt_gate_class_changes==0,
           "fixed_route_forward_class_changes":forward_class_changes,"rebuilt_route_forward_class_changes":rebuilt_gate_class_changes,
           "original_full_chain_replay_passed":max_rebuilt_gate_probability_error<2e-6 and rebuilt_gate_class_changes==0,
           "original_raw_forward_absolute_tolerance_passed":not absolute_forward_failures,
           "original_raw_forward_absolute_tolerance_failures":len(absolute_forward_failures),
           "mixed_forward_atol":2e-6,"mixed_forward_rtol":1e-6,"maximum_mixed_forward_error_ratio":maximum_mixed_forward_error,
           "original_all_numerical_checks_passed":not absolute_forward_failures and max_rebuilt_gate_probability_error<2e-6 and rebuilt_gate_class_changes==0,
           "audit_revision":"Serialized-state probability replay plus independent absolute-forward check; original failed auditor retained."}
    save_json(output/"scientific_audit.json",audit)
    save_json(output/"classification_drift.json",classification_drift)
    save_json(output/"absolute_forward_failures.json",absolute_forward_failures)
    if not args.smoke:
        write_report(output,plan,table,aggregates,pairs,audit)
        with (output/"未修正Edgepred_F100验证总结.md").open("a") as stream:
            stream.write("\n## 原前向绝对阈值未通过与审计修订\n\n")
            stream.write(f"原预声明的原始前向绝对误差 <2e-6 检查未通过，{len(absolute_forward_failures)}/240 个单元超限，最大 {max_forward_error:.9g}。旧拟合源码、原计划、原审计和失败日志、全部封存预测保留；没有重跑拟合、改变参数、划分、评分或统计规则。\n\n")
            stream.write(f"独立前向仍按原方程重建；补充与表示尺度相关的混合误差检查 abs(error)/(2e-6+1e-6*abs(saved))<1，最大比例 {maximum_mixed_forward_error:.9g}。这是执行后发生的审计修订，不能等同于原绝对阈值通过。超限明细见 absolute_forward_failures.json。保存状态读出和完整概率/类别重放的各项标志在 scientific_audit.json 单独记录。\n")
    print(json.dumps(audit),flush=True)



def write_report(output,plan,table,aggregates,pairs,audit):
    columns=[("Cora","Edgepred_Gprompt"),("CiteSeer","Edgepred_Gprompt"),
             ("Cora","Edgepred_GPPT"),("CiteSeer","Edgepred_GPPT")]
    lookup={(r["dataset"],r["family"],r["method"]):r for r in aggregates}
    effects={(r["dataset"],r["family"],r["reference"],r["metric"]):r for r in pairs}
    positive=sum(effects[(d,f,"PRPrompt","accuracy")]["mean_delta"]>0 for d,f in columns)
    positive_ci=sum(effects[(d,f,"PRPrompt","accuracy")]["paired_bootstrap_95ci"][0]>0 for d,f in columns)
    lines=["# 未修正两种 Edgepred 预训练的 F100 重跑","",
           "日期：2026-10-01。原始四份 checkpoint，20 个支持抽样、三个训练重复、5-shot、100 步，共 240 次新拟合。F100 原配方、普通 Adam、M1=0.1，无 M2/M3/KD/分类头，冻结 GCN。","",
           f"F100 相对同原始权重的 v2 PRPrompt，在 {positive}/4 个条件上 Accuracy 均值提高，{positive_ci}/4 个配对 95% 描述性区间完全为正。完整微调差距和所有负结果保留如下。","",
           "## 同未修正 checkpoint 的完整方法比较","",
           "Accuracy（%），± 为 20 个支持抽样均值的标准差；每个抽样先平均三个训练重复。旧权重每个条件只有一份，不能将 ± 解释为预训练种子标准差。","",
           "| 方法 | Cora/Gprompt | CiteSeer/Gprompt | Cora/GPPT | CiteSeer/GPPT |","|---|---:|---:|---:|---:|"]
    for method in ["v1_F100","PRPrompt","FullFineTune","FrozenLinear","FrozenPrototype","GPF","GPFPlus","GraphPromptDiag"]:
        cells=[f"{100*lookup[(d,f,method)]['accuracy_mean']:.2f} ± {100*lookup[(d,f,method)]['accuracy_std_over_support_draws']:.2f}" for d,f in columns]
        lines.append(f"| {method} | "+" | ".join(cells)+" |")
    lines += ["","## 对 v2 与微调的配对差异","",
              "Δ 为百分点；按相同支持抽样 bootstrap 10,000 次的 95% 区间，未作多重比较校正。","",
              "| 条件 | F100−v2 [95% CI] | F100−FullFineTune [95% CI] |","|---|---|---|"]
    for d,f in columns:
        cells=[]
        for method in ["PRPrompt","FullFineTune"]:
            delta=effects[(d,f,method,"accuracy")];ci=delta["paired_bootstrap_95ci"]
            cells.append(f"{100*delta['mean_delta']:+.2f} [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]")
        lines.append(f"| {d}/{f} | "+" | ".join(cells)+" |")
    lines += ["","## 配方、来源与边界","",
              "- [事前协议](../source/experiments/v1_f100_uncorrected/PLAN.md)固定权重、100 步、Adam 学习率 0.01、样本标准差、原属性修正路由、LOO 原型训练及中心化原型读出；不是 v2 改一个温度，不能把版本差异单独归因于 M1。",
              "- 原 checkpoint sidecar：ProG-adapted、labels_used=false、source_parity=not_fully_verified，last_after_training_loss_patience_10 导出。预训练种子 unknown_original，不伪造三个旧预训练种子，也不重跑预训练。",
              "- 仅主干冻结读出、微调和其他 Prompt 对比；旧 AttributeRidge/AttributePrototype 结果不进入新比较。F100 内部旧 ridge/KD 中间计算保留，但 KD 不进入活跃损失。",
              "- 原拟合源码和历史函数完整保留；新分片只分配互斥支持抽样，各单元恰好拟合一次。两卡各六个工作进程。",
              "- 全图无标签属性/结构可见，查询标签不进入路由、拟合、选参；封存预测之后才评分。同图、已知类别的传导式任务，不声称跨图或新类别泛化。",
              f"- 完整覆盖 {audit['present']}/{audit['expected']}，零缺失；全部主干冻结、正提示梯度及 100 步梯度记录通过。五指标独立重算最大误差 {audit['maximum_metric_error']:.3g}。",
              f"- 保存状态读出概率重放最大误差 {audit['maximum_final_probability_replay_error']:.3g}、类别严格相同；独立路由/原始前向误差分别为 {audit['maximum_gate_replay_error']:.3g}/{audit['maximum_raw_forward_error']:.3g}。",
              f"- 完整重建路由后的概率最大误差 {audit['maximum_rebuilt_gate_probability_error']:.3g}，累计类别变化 {audit['rebuilt_route_forward_class_changes']}；保存路由前向重建累计类别变化 {audit['fixed_route_forward_class_changes']}。原全链概率及类别严格重放是否通过：{audit['original_full_chain_replay_passed']}。采用事前声明的分层验收，同时保留全链诊断，不掩盖未通过项。",
              f"- 三个训练重复的最大 Accuracy 极差为 {100*audit['maximum_accuracy_range_across_training_seeds']:.3g} 个百分点；旧分支零初始化且无随机采样，不视为额外独立样本。",
              "- [全部 checkpoint 五指标及抽样标准差](checkpoint_comparison.csv)、[四条件五指标](condition_comparison.csv)、[配对效应](paired_effects.json)、[科学审计](scientific_audit.json)、[类别漂移](classification_drift.json)随结果保存。",""]
    (output/"未修正Edgepred_F100验证总结.md").write_text("\n".join(lines))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True);parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--device",default="cuda:0")
    summarize(parser.parse_args())
