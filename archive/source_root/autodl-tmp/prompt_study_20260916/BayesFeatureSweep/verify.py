"""Independent replay of sealed predictions, split/CV contracts, and provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

ap=argparse.ArgumentParser();ap.add_argument('root');args=ap.parse_args()
root=Path(args.root)

def digest(t):
    t=t.detach().cpu().contiguous()
    return hashlib.sha256(str((str(t.dtype),list(t.shape))).encode()+t.numpy().tobytes()).hexdigest()

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

sealed=json.loads((root/'sealed_complete.json').read_text())
for f,h in sealed['files'].items():assert sha(root/f)==h
env=json.loads((root/'environment.json').read_text())
for f,h in env['sources'].items():assert sha(f)==h,(f,'source changed')
metrics=json.loads((root/'metrics.json').read_text())
lookup={(r['dataset'],r['checkpoint'],r['seed'],r['method']):r for r in metrics}
max_acc_error=0.
max_f1_error=0.
max_noop_error=0.
count=0
diagnostics=[]
lambda_counts={}
for path in sorted((root/'sealed').rglob('*.pt')):
    obj=torch.load(path,map_location='cpu',weights_only=True)
    ds,ck,seed=obj['dataset'],obj['checkpoint'],obj['seed']
    ev=json.loads((root/f'{ds}_evaluation_truth.json').read_text())
    truth=torch.tensor(ev['truth'])
    assert digest(truth)==ev['truth_sha256']
    assert obj['query']==ev['query']
    assert digest(torch.tensor(obj['query']))==obj['query_sha256']
    assert digest(torch.tensor(obj['support']))==obj['support_sha256']
    assert not set(obj['support'])&set(obj['query'])
    if root.name=='stability':
        for old in (root.parent/'stage_a/splits'/ds).glob('*.json'):
            assert set(json.loads(old.read_text())['support'])!=set(obj['support'])
    y=dict(zip(obj['support'],obj['support_labels']))
    held=[]
    for fold in obj['folds']:
        assert not set(fold['train'])&set(fold['heldout'])
        assert set(fold['train'])|set(fold['heldout'])==set(obj['support'])
        assert len(fold['heldout'])==len(set(obj['support_labels']))
        assert digest(torch.tensor([y[k] for k in fold['train']]))==fold['label_source_sha256']
        held+=fold['heldout']
    assert sorted(held)==sorted(obj['support'])
    noop_error=float((obj['scores']['fixed']-obj['scores']['noop']).abs().max())
    max_noop_error=max(max_noop_error,noop_error)
    assert noop_error<1e-6
    assert torch.equal(obj['scores']['fixed'].argmax(1),obj['scores']['noop'].argmax(1))
    grid=env['grid'];lams=env['lambdas']
    for family in ['full','simple','label_shuffle','feature_permute']:
        j,l=min(((j,l) for j in range(len(grid)) for l in range(len(lams))),key=lambda t:(round(obj['heads'][f'{family}_{t[0]}']['cv_mse'][t[1]],12),grid[t[0]][1],grid[t[0]][0],-lams[t[1]]))
        selected=obj['heads'][f'selected_{family}']
        assert selected['selected_configuration']==j and selected['lambda']==lams[l]
        assert torch.equal(obj['scores'][f'selected_{family}'],obj['scores'][f'{family}_{j}'])
    ckpath=root.parent.parent/'provenance'/ds/f'{ck}.GCN.128hidden_dim.pth'
    assert sha(ckpath)==obj['checkpoint_sha256']
    for name,score in obj['scores'].items():
        assert torch.isfinite(score).all()
        assert digest(score)==obj['prediction_sha256'][name]
        row=lookup[ds,ck,seed,name]
        acc=float((score.argmax(1)==truth).double().mean())
        max_acc_error=max(max_acc_error,abs(acc-row['accuracy']))
        assert abs(acc-row['accuracy'])<1e-12
        pred=score.argmax(1)
        f1=[]
        for k in range(score.shape[1]):
            den=int((pred==k).sum()+(truth==k).sum())
            f1.append(2*int(((pred==k)&(truth==k)).sum())/den if den else 0.)
        max_f1_error=max(max_f1_error,abs(sum(f1)/len(f1)-row['macro_f1']))
        assert max_f1_error<1e-12
        head=obj['heads'][name]
        if 'lambda' in head:
            lams=env['lambdas'];loss=head['cv_mse']
            best=min(range(len(loss)),key=lambda j:(round(loss[j],12),-lams[j]))
            assert head['lambda']==lams[best]
            if name in ['main','fixed']:
                key=f'{ds}/{ck}/{name}/{head["lambda"]}'
                lambda_counts[key]=lambda_counts.get(key,0)+1
        count+=1
    diagnostics.append({'dataset':ds,'checkpoint':ck,'seed':seed,**obj['diagnostics']})
for f in root.glob('*_audit.json'):
    obj=json.loads(f.read_text());assert obj['before']==obj['after'] and obj['frozen']
for f in root.glob('*_data.json'):
    assert json.loads(f.read_text())['non_train_labels_poisoned']
assert count==sealed['records']==len(metrics)==2*4*env['args']['seed_count']*31
out={'passed':True,'records_checked':count,'max_accuracy_replay_error':max_acc_error,
    'max_macro_f1_replay_error':max_f1_error,'max_noop_score_error':max_noop_error,
    'sealed_file_hashes_match':True,'source_hashes_match':True,'cv_lambda_selection_verified':True,'joint_configuration_selection_verified':True,'checkpoint_snapshot_hashes_verified':True,
    'inner_folds_label_sources_verified':True,'zero_prompt_scores_within_1e_6':True,'zero_prompt_predictions_equal':True,
    'backbone_freeze_verified':True,'nontrain_label_poisoning_recorded':True,
    'lambda_counts':lambda_counts,'diagnostics':diagnostics,
    'limitation':'Replay proves saved predictions and implementation contracts, not effectiveness or independent generalization.'}
(root/'verification.json').write_text(json.dumps(out,indent=2)+'\n')
print('PASS',count,'records; accuracy replay error',max_acc_error)
