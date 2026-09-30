"""Bounded support-selected extension; original EBFP implementation stays immutable."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'BayesFeaturePrompt'))
import importlib.util
spec=importlib.util.spec_from_file_location('original_ebfp',Path(__file__).resolve().parents[1]/'BayesFeaturePrompt/experiment.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
Base=mod.Experiment
from run import *

GRID=[(5.,.25),(5.,.5),(5.,1.),(1.,.5),(20.,.5)]

class Experiment(Base):
    def all_features(self,s,y,c):
        zs=dict(self.constants)
        zs['noop']=self.encode(self.x)
        for family in ['full','simple','label_shuffle','feature_permute']:
            yy=y
            if family=='label_shuffle':
                yy=y[torch.randperm(len(y),device=y.device,generator=generator(11))]
            for j,(k,a) in enumerate(GRID):
                g,_,_=self.weights(s,yy,c,kappa=k,debias=family!='simple')
                g=1+(g-1)*(a/.5)
                if family=='feature_permute':
                    g=g[torch.randperm(len(g),device=g.device,generator=generator(11))]
                zs[f'{family}_{j}']=self.encode(self.transform(g))
        return zs

    def predict(self,s,y,c):
        losses=None;folds=[]
        for fold in range(5):
            val=torch.stack([s[y==k][fold] for k in range(c)])
            keep=~torch.isin(s,val);ts,ty=s[keep],y[keep]
            zs=self.all_features(ts,ty,c)
            if losses is None: losses={n:[0.]*len(LAMBDAS) for n in zs}
            target=torch.eye(c,device=s.device,dtype=torch.double)
            for n,z in zs.items():
                for j,lam in enumerate(LAMBDAS):
                    losses[n][j]+=float((ridge_fit(z,ts,ty,c,lam)[val]-target).square().mean())/5
            folds.append({'train':ts.cpu().tolist(),'heldout':val.cpu().tolist(),'label_source_sha256':digest(ty),'disjoint':True,'prompt_rebuilt':True})
        zs=self.all_features(s,y,c);outputs={};heads={}
        for n,z in zs.items():
            j=min(range(len(LAMBDAS)),key=lambda j:(round(losses[n][j],12),-LAMBDAS[j]))
            outputs[n]=ridge_fit(z,s,y,c,LAMBDAS[j])
            heads[n]={'lambda':LAMBDAS[j],'cv_mse':losses[n],'support_mse':float((outputs[n][s]-F.one_hot(y,c)).square().mean())}
        for family in ['full','simple','label_shuffle','feature_permute']:
            j,l=min(((j,l) for j in range(len(GRID)) for l in range(len(LAMBDAS))),key=lambda t:(round(losses[f'{family}_{t[0]}'][t[1]],12),GRID[t[0]][1],GRID[t[0]][0],-LAMBDAS[t[1]]))
            name=f'{family}_{j}'
            outputs[f'selected_{family}']=outputs[name]
            heads[f'selected_{family}']={**heads[name],'selected_configuration':j,'kappa':GRID[j][0],'amplitude':GRID[j][1]}
        assert torch.allclose(outputs['noop'],outputs['fixed'],atol=1e-6,rtol=1e-5)
        assert torch.equal(outputs['noop'].argmax(1),outputs['fixed'].argmax(1))
        return outputs,heads,{'noop_max_error':float((outputs['noop']-outputs['fixed']).abs().max())},folds
