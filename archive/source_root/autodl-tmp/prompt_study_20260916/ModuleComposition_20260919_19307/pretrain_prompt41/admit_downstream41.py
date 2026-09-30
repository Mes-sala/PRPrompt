"""Unscored readout/adapter parity on baseline and prompted representations."""
from downstream41_common import *
import time


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    reports=[];start=time.time()
    oldcomplete=json.loads((BASE/'four33/complete.json').read_text())
    for ds in ['ENZYMES','MUTAG','PROTEINS']:
        graphs,dim=graphs41(ds);ep=episode(episodes(ds)[0]);static=static41(ds,ep)
        choices=[('random_s101','standard',BASE/'repair31_training'/ds/'random_s101/encoder.pth'),
                 ('DGI_both_s101','standard',BASE/'repair31_training'/ds/'DGI_both_s101/encoder.pth'),
                 ('GraphMAE2_s101','standard',ROOT/'masked_training'/ds/'GraphMAE2_s101/encoder.pth'),
                 ('MultiG_smoke','multig',ROOT/'multig_smoke'/ds/'GraphMultiGprompt_repaired_s101/encoder.pth')]
        for name,kind,path in choices:
            m=model41(dim,path,kind);initial=fingerprint(m);blocks=pooled41(m,graphs)
            out,err=readout41(kernels41(blocks),static,ep);old_match=None
            if name in ['random_s101','DGI_both_s101']:
                rec=oldcomplete['records'][ds+'/'+name]
                with np.load(BASE/rec['file']) as z:
                    old=z['predictions'][0,rec['methods'].index('late_mean__WAHM')]
                assert np.array_equal(np.round(out['four'],10).argmax(-1),old),(ds,name,'old baseline replay')
                old_match=True
            method='GPF-plus+Gprompt';f,c,head=prompts41(dim,int(ep['support_labels'].max())+1,method)
            m.float().cuda();f.cuda();c.cuda();head.cuda();assert fingerprint(m)==initial
            b=Batch.from_data_list([graphs[int(i)] for i in ep['support']]).to('cuda')
            y=torch.tensor(ep['support_labels'],device='cuda',dtype=torch.long)
            opt=torch.optim.Adam([*f.parameters(),*c.parameters()],lr=.001,weight_decay=.0005)
            fp_before=fingerprint(f);cp_before=fingerprint(c)
            losses=[train_step41(m,b,y,f,c,head,opt) for _ in range(3)]
            assert fingerprint(m)==initial and (fingerprint(f)!=fp_before or fingerprint(c)!=cp_before)
            prompted=pooled41(m,graphs,f,c);o2,e2=readout41(kernels41(prompted),static,ep)
            assert any(not torch.equal(a,b) for a,b in zip(blocks,prompted))
            reports.append({'dataset':ds,'encoder':name,'kind':kind,'checkpoint_sha256':digest(path),'old_four_predictions_equal':old_match,'max_algebra_error':max(err,e2),'frozen_encoder':True,'support_losses':losses,'prompt_changed_features':True,'query_labels_read':False})
            print('DOWNSTREAM41_ADMIT',ds,name,flush=True)
    with (ROOT/'downstream_admission.json').open('x') as f:
        json.dump({'status':'passed','cases':reports,'count':len(reports),'seconds':time.time()-start,'scope':'Unscored prediction and adapter parity; MultiG uses its separate one-step admission checkpoint, not production classification evidence.','source_sha256':digest(Path(__file__)),'common_sha256':digest(ROOT/'downstream41_common.py')},f,indent=2)


if __name__=='__main__':main()
