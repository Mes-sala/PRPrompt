import torch
class MatrixPreconditioner(torch.optim.Optimizer):
    def __init__(self,params,lr,mode='shampoo',epsilon=1e-4):
        assert mode in ['shampoo','diagonal','adagrad']
        super().__init__(params,dict(lr=lr,mode=mode,epsilon=epsilon))
        self.diagnostics=[]
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:continue
                g=p.grad.detach().cpu().double();state=self.state[p];eps=group['epsilon'];mode=group['mode']
                if not state:
                    state['L']=torch.eye(g.shape[0],dtype=torch.float64)*eps
                    state['R']=torch.eye(g.shape[1],dtype=torch.float64)*eps
                    state['V']=torch.full_like(g,eps)
                if mode=='adagrad':
                    state['V']+=g.square();direction=g/state['V'].sqrt();low=float(state['V'].min());high=float(state['V'].max())
                else:
                    state['L']+=g@g.T;state['R']+=g.T@g
                    if mode=='diagonal':
                        direction=state['L'].diag().pow(-.25)[:,None]*g*state['R'].diag().pow(-.25)[None,:]
                        low=min(float(state['L'].diag().min()),float(state['R'].diag().min()));high=max(float(state['L'].diag().max()),float(state['R'].diag().max()))
                    else:
                        vl,ul=torch.linalg.eigh(state['L']);vr,ur=torch.linalg.eigh(state['R']);assert vl.min()>0 and vr.min()>0
                        lp=(ul*vl.pow(-.25))@ul.T;rp=(ur*vr.pow(-.25))@ur.T;direction=lp@g@rp
                        low=min(float(vl.min()),float(vr.min()));high=max(float(vl.max()),float(vr.max()))
                assert torch.isfinite(direction).all()
                p.add_(direction.to(device=p.device,dtype=p.dtype),alpha=-group['lr'])
                self.diagnostics.append(dict(min_eigenvalue=low,max_eigenvalue=high,update_norm=float(direction.norm())*group['lr']))
