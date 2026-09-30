from downstream41_common import *
from run_downstream41 import atomic_npz
import time

def main():
 cfg=json.loads((ROOT/'downstream_freeze.json').read_text());out=ROOT/'no_prompt_retest_20260921';out.mkdir(exist_ok=False)
 for n,h in cfg['source_sha256'].items():assert digest(ROOT/n)==h,n
 for n,h in cfg['inputs'].items():assert digest(BASE/n)==h,n
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 start=time.time();done=[];methods=['four','HM','WA','H','M']
 for cell in cfg['cells']:
  ds,enc=cell['dataset'],cell['encoder'];path=BASE/cell['checkpoint'];assert digest(path)==cell['checkpoint_sha256']
  graphs,dim=graphs41(ds);eps=[episode(episodes(ds)[i]) for i in cfg['episode_indices']];model=model41(dim,path,cell['kind']);blocks=pooled41(model,graphs);ks=kernels41(blocks);scores=[];errors=[]
  for ep in eps:
   score,err=readout41(ks,static41(ds,ep),ep);scores.append(np.stack([score[m] for m in methods]));errors.append(err)
  scores=np.stack(scores);pred=np.round(scores,10).argmax(-1).astype(np.int16)
  original=ROOT/'downstream'/ds/enc/'no_prompt';rec=json.loads((original/'record.json').read_text());assert digest(original/'sealed.npz')==rec['sealed_sha256']
  with np.load(original/'sealed.npz') as z:
   assert np.array_equal(scores,z['scores']),(ds,enc,'scores')
   assert np.array_equal(pred,z['predictions']),(ds,enc,'predictions')
   assert np.array_equal(np.stack([e['support'] for e in eps]),z['supports']) and np.array_equal(eps[0]['query'],z['query'])
  folder=out/ds/enc;folder.mkdir(parents=True);atomic_npz(folder/'sealed.npz',scores=scores,predictions=pred,supports=np.stack([e['support'] for e in eps]),query=eps[0]['query'])
  r={'dataset':ds,'encoder':enc,'checkpoint_sha256':digest(path),'sealed_sha256':digest(folder/'sealed.npz'),'original_sealed_sha256':rec['sealed_sha256'],'exact_replay':True,'methods':methods,'max_algebra_error':max(errors),'prompt_used':False,'query_labels_read':False}
  write(folder/'record.json',r);done.append(r);print('NO_PROMPT_RETEST',len(done),42,ds,enc,flush=True)
 write(out/'complete.json',{'status':'42_cells_exact_no_prompt_replay_passed','count':len(done),'records':done,'source_sha256':digest(Path(__file__)),'freeze_sha256':digest(ROOT/'downstream_freeze.json'),'seconds':time.time()-start,'query_labels_read':False})
if __name__=='__main__':main()
