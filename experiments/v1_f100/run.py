"""Fit historical F100; support labels only; seal before evaluation."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from prprompt.model import TwoLayerGCN, normalized_adjacency
from prprompt.train import state_hash, tensor_hash
from prprompt.protocol import load_dataset, sample_support, sha_file, save_json, seal_predictions, evaluate_sealed
from experiments.v1_f100.historical import load, fit_f100


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_plan(path):
    plan = json.loads(Path(path).read_text())
    identity_key = "document_sha256" if "document_sha256" in plan else "plan_sha256"
    assert digest({k:v for k,v in plan.items() if k!=identity_key}) == plan[identity_key]
    for p,h in plan["source"].items():
        assert sha_file(ROOT/p)==h,p
    for reference in plan["references"].values():
        assert sha_file(reference["plan_file"])==reference["plan_file_sha256"]
        assert sha_file(reference["manifest_file"])==reference["manifest_file_sha256"]
    return plan


def run(args):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    plan=read_plan(args.plan);job=plan["jobs"][args.job_index]
    output=args.plan.parent/("smoke" if args.smoke else "results")/job["id"]
    output.mkdir(parents=True,exist_ok=True)
    lock=output/"running.lock"
    with lock.open("x") as stream:stream.write(str(os.getpid()))
    try:
        assert sha_file(job["checkpoint"])==job["checkpoint_sha256"]
        x,edges,labels,pool,query,classes=load_dataset(job["dataset"],plan["data_root"])
        graph_hash=digest({"x":tensor_hash(x),"edges":tensor_hash(edges)})
        device=torch.device(args.device);x,edges=x.to(device),edges.to(device)
        model=TwoLayerGCN(job["input_dim"],128,128).to(device).eval()
        model.load_state_dict(torch.load(job["checkpoint"],map_location="cpu",weights_only=True),strict=True)
        model.requires_grad_(False);before=state_hash(model)
        adjacency=normalized_adjacency(edges,len(x),x.dtype);old=load()
        with torch.no_grad():
            u=model.conv_layers[0](x,edges).detach()
            r=torch.sparse.mm(adjacency,torch.sparse.mm(adjacency,x)).detach()
            scale=u.std(0).clamp_min(1e-6)
            filters=old["make_filters"](x,edges)
        split_seeds=plan["split_seeds"][:1] if args.smoke else plan["split_seeds"]
        train_seeds=plan["train_seeds"][:1] if args.smoke else plan["train_seeds"]
        steps=2 if args.smoke else 100;completed=0
        for split_seed in split_seeds:
            support=sample_support(pool,labels[pool],classes,5,split_seed)
            sy=labels[support];assert not torch.isin(support,query).any()
            assert torch.all(torch.bincount(sy,minlength=classes)==5)
            for seed in train_seeds:
                folder=output/"5"/str(split_seed)/str(seed)/"F100"
                folder.mkdir(parents=True,exist_ok=True)
                done=folder/"metrics.json"
                if done.exists():
                    row=json.loads(done.read_text());assert row["plan_sha256"]==plan["plan_sha256"]
                    for name,field in [("state.pt","state_sha256"),("selection.json","selection_sha256"),("predictions.npz","predictions_sha256")]:
                        assert sha_file(folder/name)==row[field],str(folder/name)
                    completed+=1;continue
                baseline=Path(plan["base"])/job["id"]/"5"/str(split_seed)/str(seed)/"PRPrompt"
                ref=json.loads((baseline/"metrics.json").read_text())
                assert ref["support_sha256"]==tensor_hash(support) and ref["query_sha256"]==tensor_hash(query)
                assert ref["support_labels_sha256"]==tensor_hash(sy) and ref["graph_sha256"]==graph_hash
                began=time.perf_counter()
                z,prompt,trace=fit_f100(u,r,adjacency,model,edges,support.to(device),sy.to(device),classes,seed,steps,filters)
                assert not trace["kd_active"] and trace["optimizer82"]=="adam"
                assert trace["trainable_parameters"]==classes*128
                gradients=prompt["optimizer_gradients"]
                assert gradients.shape==(steps,classes,128) and torch.isfinite(gradients).all()
                assert float(gradients.abs().max())>0 and torch.count_nonzero(prompt["tokens"])>0
                assert before==state_hash(model) and all(p.grad is None for p in model.parameters())
                with torch.no_grad():
                    center=z.mean(0,keepdim=True);centered=z-center
                    scores=old["logits"](centered,support.to(device),sy.to(device),classes)
                state={"prompt":prompt,"raw_embedding":z.cpu(),"center":center.cpu(),
                       "centered_embedding":centered.cpu(),"scale":scale.cpu(),
                       "support":support,"support_labels":sy,"query":query}
                torch.save(state,folder/"state.pt")
                selection={"selection":"fixed_historical_F100_recipe","learning_rate":.01,
                           "steps":steps,"weight_decay":0,"adam_epsilon":1e-8,
                           "prompt_penalty":.001,"M1_temperature":.1,"M2":False,"M3":False,
                           "loo_prototype_training":True,"query_labels_used":False,
                           "legacy_ridge_lambda_unused_in_active_loss":trace.get("ridge_lambda")}
                save_json(folder/"selection.json",selection)
                metadata={"plan_sha256":plan["plan_sha256"],"method":"v1_F100","dataset":job["dataset"],
                          "family":job["family"],"checkpoint_id":job["id"],"pretrain_seed":job["pretrain_seed"],
                          "checkpoint_sha256":job["checkpoint_sha256"],"split_seed":split_seed,"train_seed":seed,
                          "graph_sha256":graph_hash,"support_labels_sha256":tensor_hash(sy),"smoke":args.smoke}
                signature=digest(metadata)
                seal=seal_predictions(folder,scores,support,query.to(device),dict(metadata,signature=signature))
                # First query-label indexing occurs after scores and the seal are saved.
                metrics=evaluate_sealed(folder,labels[query])
                audit={"backbone_before":before,"backbone_after":state_hash(model),"backbone_frozen":True,
                       "query_labels_received":False,"M2_active":False,"M3_active":False,"head_parameters":0,
                       "prompt_parameters":classes*128,"prompt_gradient_max":float(gradients.abs().max()),
                       "gradient_steps":len(gradients),"kd_active":False,"adam_actual_epsilon":1e-8,
                       "legacy_trace_optimizer_epsilon_is_not_adam_setting":trace["optimizer_epsilon"]}
                row=dict(metadata,signature=signature,predictions_sha256=seal["predictions_sha256"],
                         state_sha256=sha_file(folder/"state.pt"),selection_sha256=sha_file(folder/"selection.json"),
                         support_sha256=seal["support_sha256"],query_sha256=seal["query_sha256"],
                         source_reference_metrics_sha256=sha_file(baseline/"metrics.json"),
                         metrics=metrics,audit=audit,trace=trace,seconds=time.perf_counter()-began,
                         environment={"torch":str(torch.__version__),"cuda":torch.version.cuda,"device":str(device)})
                save_json(done,row);completed+=1
                print(json.dumps({"job":job["id"],"complete":completed,"expected":len(split_seeds)*len(train_seeds),"seconds":row["seconds"]}),flush=True)
        save_json(output/"completion.json",{"plan_sha256":plan["plan_sha256"],"units":completed,"smoke":args.smoke})
    finally:lock.unlink()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--job-index",type=int,required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--smoke",action="store_true")
    run(parser.parse_args())
