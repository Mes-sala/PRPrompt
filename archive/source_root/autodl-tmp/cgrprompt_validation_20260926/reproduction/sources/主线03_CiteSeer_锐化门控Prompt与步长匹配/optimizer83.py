import torch
class GraftedMatrix(torch.optim.Optimizer):
    def __init__(self,params,lr=.01,mode='shampoo',epsilon=1e-4):
        assert mode in ['shampoo','diagonal','gradient','adam']
        super().__init__(params,dict(lr=lr,mode=mode,epsilon=epsilon));self.diagnostics=[]
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:continue
                g=p.grad.detach().cpu().double();state=self.state[p];mode=group['mode']
                if not state:
                    state.update(L=torch.eye(g.shape[0],dtype=g.dtype)*group['epsilon'],R=torch.eye(g.shape[1],dtype=g.dtype)*group['epsilon'],m=torch.zeros_like(g),v=torch.zeros_like(g),step=0)
                state['step']+=1;t=state['step'];state['m'].mul_(.9).add_(g,alpha=.1);state['v'].mul_(.999).addcmul_(g,g,value=.001)
                adam=(state['m']/(1-.9**t))/((state['v']/(1-.999**t)).sqrt()+1e-8)
                state['L']+=g@g.T;state['R']+=g.T@g
                if mode=='shampoo':
                    vl,ul=torch.linalg.eigh(state['L']);vr,ur=torch.linalg.eigh(state['R']);assert vl.min()>0 and vr.min()>0
                    direction=((ul*vl.pow(-.25))@ul.T)@g@((ur*vr.pow(-.25))@ur.T)
                elif mode=='diagonal':direction=state['L'].diag().pow(-.25)[:,None]*g*state['R'].diag().pow(-.25)[None,:]
                elif mode=='gradient':direction=g
                else:direction=adam
                dn=direction.norm();an=adam.norm()
                # A zero candidate direction cannot borrow a nonzero magnitude.
                # Explicitly preserve a no-op, recording whether momentum remained.
                update=direction*(an/dn) if dn>0 else torch.zeros_like(direction)
                assert torch.isfinite(update).all()
                if dn>0:assert torch.allclose(update.norm(),an,atol=1e-12,rtol=1e-10)
                p.add_(update.to(device=p.device,dtype=p.dtype),alpha=-group['lr'])
                self.diagnostics.append(dict(direction_norm=float(dn),adam_norm=float(an),update_norm=float(update.norm())*group['lr'],zero_direction=bool(dn==0)))
