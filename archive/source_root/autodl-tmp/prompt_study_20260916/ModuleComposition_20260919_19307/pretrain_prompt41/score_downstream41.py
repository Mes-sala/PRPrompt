"""Score only the complete sealed grid; bind every score file and paired control."""
from downstream41_common import *
from score_joint22 import metrics
import csv,time

FIELDS=['accuracy','macro_f1','macro_recall']


def main():
    start=time.time();folder=ROOT/'downstream';cfg=json.loads((ROOT/'downstream_freeze.json').read_text())
    complete=json.loads((folder/'complete.json').read_text())
    assert complete['status']=='sealed_unscored' and complete['count']==cfg['total_jobs']==5040
    assert complete['freeze_sha256']==digest(ROOT/'downstream_freeze.json')
    assert len(cfg['cells'])==42 and cfg['episode_indices']==list(range(20))
    assert not (folder/'analysis.json').exists(),'Do not overwrite final analysis'
    rows=[];values={};file_shas={};truths={};original_random={};old_best={};episode_cache={};old_single={}
    for ds in ['ENZYMES','MUTAG','PROTEINS']:
        labels_path=BASE/'graph_assets/raw_datasets'/ds/'raw'/(ds+'_graph_labels.txt')
        raw=np.loadtxt(labels_path,dtype=np.int64);_,labels=np.unique(raw,return_inverse=True)
        truths[ds]=labels;file_shas[str(labels_path.relative_to(BASE))]=digest(labels_path)
        eps=[episode(e) for e in episodes(ds)];episode_cache[ds]=eps
        for ep in eps:assert np.array_equal(labels[ep['support']],ep['support_labels'])
    old_rows=list(csv.DictReader((BASE/'four33/metrics.csv').open()))
    file_shas['four33/metrics.csv']=digest(BASE/'four33/metrics.csv')
    for ds in truths:
        old_by={}
        for r in old_rows:
            if r['dataset']==ds and r['method']=='late_mean__WAHM':old_by.setdefault(r['encoder'],{})[int(r['support_seed'])]=[float(r[k]) for k in FIELDS]
        seq=[int(e['seed']) for e in episodes(ds)]
        old_single[ds]=np.array([old_by['random_s101'][seed] for seed in seq])
        original_random[ds]=sum(np.array([old_by[f'random_s{s}'][seed] for seed in seq]) for s in [101,211,307])/3
        best=min(old_by,key=lambda n:(-np.mean([v[0] for v in old_by[n].values()]),n))
        old_best[ds]=(best,np.array([old_by[best][seed] for seed in seq]))

    def add(ds,enc,method,lr,step,readout,ei,pred):
        ep=episode_cache[ds][ei];truth=truths[ds][ep['query']]
        result=metrics(truth,pred,int(truths[ds].max())+1)
        rows.append({'dataset':ds,'encoder':enc,'prompt':method,'lr':lr,'step':step,'readout':readout,'episode':ei,'support_seed':episodes(ds)[ei]['seed'],**result})
        key=(ds,enc,method,str(lr),int(step),readout)
        values.setdefault(key,{})[ei]=np.array([result[k] for k in FIELDS])

    for cell in cfg['cells']:
        ds,enc=cell['dataset'],cell['encoder'];base=folder/ds/enc
        rec=json.loads((base/'no_prompt/record.json').read_text())
        assert rec['freeze_sha256']==complete['freeze_sha256']
        path=base/'no_prompt/sealed.npz';assert digest(path)==rec['sealed_sha256'];file_shas[str(path.relative_to(BASE))]=digest(path)
        with np.load(path) as z:
            assert np.array_equal(np.round(z['scores'],10).argmax(-1),z['predictions'])
            for ei in range(20):
                ep=episode_cache[ds][ei];assert np.array_equal(z['supports'][ei],ep['support']) and np.array_equal(z['query'],ep['query'])
                for mi,readout in enumerate(rec['methods']):add(ds,enc,'none',0,0,readout,ei,z['predictions'][ei,mi])
        for ei in range(20):
            for method in cfg['prompt_methods']:
                zero=None
                for lr in cfg['learning_rates']:
                    job=base/f'episode{ei:02d}__{method}__lr{lr:g}'
                    r=json.loads((job/'record.json').read_text());assert r['freeze_sha256']==complete['freeze_sha256'] and r['checkpoint_sha256']==cell['checkpoint_sha256']
                    path=job/'sealed.npz';assert digest(path)==r['sealed_sha256'] and digest(job/'prompt_states.pt')==r['state_sha256']
                    file_shas[str(path.relative_to(BASE))]=digest(path);file_shas[str((job/'record.json').relative_to(BASE))]=digest(job/'record.json')
                    with np.load(path) as z:
                        ep=episode_cache[ds][ei];assert np.array_equal(z['support'],ep['support']) and np.array_equal(z['support_labels'],ep['support_labels']) and np.array_equal(z['query'],ep['query'])
                        pred=np.round(z['scores'],10).argmax(-1);pred[:,-1]=z['scores'][:,-1].argmax(-1)
                        assert np.array_equal(pred,z['predictions'])
                        if zero is None:zero=z['scores'][0].copy()
                        else:assert np.array_equal(zero,z['scores'][0]),'Same untrained Prompt across learning rates'
                        for si,step in enumerate(r['steps']):
                            for mi,readout in enumerate(r['methods']):add(ds,enc,method,lr,step,readout,ei,pred[si,mi])
        print('SCORE41',ds,enc,flush=True)
    assert len(rows)==94920
    arrays={k:np.stack([v[i] for i in range(20)]) for k,v in values.items()}
    for ds in truths:assert np.allclose(arrays[(ds,'random_s101','none','0',0,'four')],old_single[ds],atol=1e-10,rtol=0),'Old random four replay'
    bootstrap=np.random.default_rng(19311441).integers(0,20,(10000,20))
    def pair(a,b):
        d=a-b;v=d[:,0]
        return {'mean_pp':float(v.mean()),'support_differences_pp':v.tolist(),'support_ci95':np.percentile(v[bootstrap].mean(1),[2.5,97.5]).tolist(),'nonnegative_supports':int((v>=-1e-10).sum()),'macro_f1_pp':float(d[:,1].mean()),'macro_recall_pp':float(d[:,2].mean())}
    def gate(p):return p['mean_pp']>=1 and p['support_ci95'][0]>0 and p['nonnegative_supports']>=16 and p['macro_f1_pp']>=0 and p['macro_recall_pp']>=0
    results={};leads=[]
    for key,arr in arrays.items():
        ds,enc,method,lr,step,readout=key
        if method=='none' or step==0 or readout!='four':continue
        pairs={
            'same_encoder_no_prompt':pair(arr,arrays[(ds,enc,'none','0',0,'four')]),
            'untrained_prompt':pair(arr,arrays[(ds,enc,method,lr,0,'four')]),
            'current_random_s101':pair(arr,arrays[(ds,'random_s101','none','0',0,'four')]),
            'old_random_three_seed_metric_average':pair(arr,original_random[ds]),
            'old_best_development_four':pair(arr,old_best[ds][1])}
        if enc.startswith('GraphMultiGprompt'):pairs['same_architecture_random']=pair(arr,arrays[(ds,'GraphMultiGprompt_random_s101','none','0',0,'four')])
        name='/'.join(map(str,key[:-1]));passed=gate(pairs['same_encoder_no_prompt']) and gate(pairs['current_random_s101'])
        results[name]={'means':dict(zip(FIELDS,arr.mean(0).tolist())),'pairs':pairs,'old_best_encoder':old_best[ds][0],'passes_initial_screen':passed,'prompt_beats_untrained_mean':pairs['untrained_prompt']['mean_pp']>0}
        if passed:leads.append(name)
    metric_path=folder/'metrics.csv'
    with metric_path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(folder/'analysis.json',{'status':'complete_development_comparison','rows':len(rows),'comparisons':results,'leads':leads,'metrics_sha256':digest(metric_path),'freeze_sha256':digest(ROOT/'downstream_freeze.json'),'source_sha256':digest(Path(__file__)),'query_unseen':False,'multiple_testing_corrected':False,'scope':'Fixed shared-query paired support bootstrap; correlated supports, retrospective candidates require subsequent confirmation.','seconds':time.time()-start})
    write(folder/'score_audit.json',{'status':'all_sealed_scores_and_episode_identities_verified','file_sha256':file_shas,'rows':len(rows),'source_sha256':digest(Path(__file__)),'model_inference_repeated':False,'bootstrap_seed':19311441,'bootstrap_samples':10000})


if __name__=='__main__':main()
