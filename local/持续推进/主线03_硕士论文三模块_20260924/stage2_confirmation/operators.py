# Frozen explicit ablation operators; historical sources untouched.
import torch
import torch.nn.functional as F
from run_round53_focal import sha,thash,act,logits,ridge_teacher,rectified_teacher,smooth_probs,focal_loss
def a1_ablate(u, r, adj, model, edges, support, sy, classes, seed, method, steps, filters=None, gamma=0., mode="focal", ablation="none"):
    torch.manual_seed(seed)
    if 'ref' in method:
        teacher_scores, ridge_info = rectified_teacher(r, support, sy, classes, method, seed)
        teacher = teacher_scores.softmax(-1).detach()
    elif 'ridge' in method:
        teacher_scores, ridge_info = ridge_teacher(r, support, sy, classes)
        teacher = teacher_scores.softmax(-1).detach()
    else:
        teacher = logits(r, support, sy, classes).softmax(-1).detach()
        ridge_info = {}
    if "plain" in method:
        teacher = smooth_probs(teacher, filters["plain"]).detach()
        ridge_info["teacher_filter"] = "plain"
    gate = teacher - teacher.mean(0, keepdim=True)
    if method.endswith('perm'):
        gate = gate[torch.randperm(len(gate), device=gate.device)]
    gate = torch.sparse.mm(adj, gate).detach()
    if ablation=='permgate':
        gen_gate=torch.Generator(device=u.device).manual_seed(seed+240924)
        gate=gate[torch.randperm(len(gate),generator=gen_gate,device=u.device)]
    scale = u.std(0).clamp_min(1e-6).detach()
    tokens = torch.nn.Parameter(torch.zeros(classes, u.shape[1], device=u.device))
    tokens2 = torch.nn.Parameter(torch.zeros_like(tokens))
    params = [tokens, tokens2] if method.endswith('deep') else [tokens]
    hbase = act(u).detach()
    zbase = model.conv_layers[1](hbase, edges).detach()
    hscale = hbase.std(0).clamp_min(1e-6).detach()
    zscale = zbase.std(0).clamp_min(1e-6).detach()
    amp = 10.0 if '10' in method else (3.0 if '3' in method else 1.0)
    def forward():
        if method.endswith('late'):
            return model.conv_layers[1](hbase + amp * (gate @ tokens.tanh()) * hscale, edges)
        if method.endswith('output'):
            return zbase + amp * (gate @ tokens.tanh()) * zscale
        if method.endswith('deep'):
            hp = act(u + (amp / 2**0.5) * (gate @ tokens.tanh()) * scale)
            hp = hp + (amp / 2**0.5) * (gate @ tokens2.tanh()) * hscale
            return model.conv_layers[1](hp, edges)
        hp = act(u + amp * (gate @ tokens.tanh()) * scale)
        return model.conv_layers[1](hp, edges)
    with torch.no_grad():
        noop = (F.normalize(forward(), dim=1) - F.normalize(zbase, dim=1)).abs().max().item()
        assert noop < 2e-5
    if method.startswith('token_random'):
        with torch.no_grad():
            tokens.normal_(0, 0.3)
        return forward().detach(), {'tokens': tokens.detach().cpu(), 'amplitude': amp}, {}
    gen=torch.Generator(device=u.device).manual_seed(seed+530000)
    scales=[]
    opt = torch.optim.Adam(params, lr=0.01)
    hist = []
    for step in range(steps):
        opt.zero_grad()
        z = forward()
        loo=logits(z,support,sy,classes,loo=True)
        ce=F.cross_entropy(loo,sy)
        supervised,coeff=focal_loss(loo,sy,gamma,mode,gen)
        pt=loo.softmax(-1).gather(1,sy[:,None]).flatten()
        scales.append(dict(mean=coeff.mean().item(),minimum=coeff.min().item(),maximum=coeff.max().item()))
        kd = F.kl_div(logits(z, support, sy, classes).log_softmax(-1), teacher, reduction='batchmean')
        loss = supervised + (kd if 'kd' in method and ablation!='nokd' else 0) + 0.001 * sum(p.square().mean() for p in params)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None for p in model.parameters())
        opt.step()
        if step in (0, steps-1):
            hist.append({'step': step, 'loss': loss.item(), 'ce': ce.item(), 'kd': kd.item(), 'gradient_norm': tokens.grad.norm().item(),'supervised':supervised.item(),'mean_pt':pt.mean().item()})
    return forward().detach(), {'tokens': tokens.detach().cpu(), 'tokens2': tokens2.detach().cpu(), 'amplitude': amp}, {'focal_gamma':gamma,'focal_mode':mode,'coefficient_history':scales,'history': hist, 'prompt_noop_norm_error': noop, 'trainable_parameters': sum(p.numel() for p in params), **ridge_info}

def a4_ablate(u, r, adj, model, edges, support, sy, classes, seed, method, steps, filters=None, vatmode="none", epsilon=0., ablation="none"):
    torch.manual_seed(seed)
    if 'ref' in method:
        teacher_scores, ridge_info = rectified_teacher(r, support, sy, classes, method, seed)
        teacher = teacher_scores.softmax(-1).detach()
    elif 'ridge' in method:
        teacher_scores, ridge_info = ridge_teacher(r, support, sy, classes)
        teacher = teacher_scores.softmax(-1).detach()
    else:
        teacher = logits(r, support, sy, classes).softmax(-1).detach()
        ridge_info = {}
    if "plain" in method:
        teacher = smooth_probs(teacher, filters["plain"]).detach()
        ridge_info["teacher_filter"] = "plain"
    gate = teacher - teacher.mean(0, keepdim=True)
    if method.endswith('perm'):
        gate = gate[torch.randperm(len(gate), device=gate.device)]
    gate = torch.sparse.mm(adj, gate).detach()
    scale = u.std(0).clamp_min(1e-6).detach()
    tokens = torch.nn.Parameter(torch.zeros(classes, u.shape[1], device=u.device))
    tokens2 = torch.nn.Parameter(torch.zeros_like(tokens))
    params = [tokens, tokens2] if method.endswith('deep') else [tokens]
    hbase = act(u).detach()
    zbase = model.conv_layers[1](hbase, edges).detach()
    hscale = hbase.std(0).clamp_min(1e-6).detach()
    zscale = zbase.std(0).clamp_min(1e-6).detach()
    amp = 10.0 if '10' in method else (3.0 if '3' in method else 1.0)
    perturb_scale=hscale if method.endswith('late') else scale
    def forward(delta=None):
        noise=0 if delta is None else delta*perturb_scale
        if method.endswith('late'):
            hp=hbase+amp*(gate@tokens.tanh())*hscale+noise
        else:
            hp=act(u+amp*(gate@tokens.tanh())*scale+noise)
        return model.conv_layers[1](hp,edges)
    with torch.no_grad():
        noop = (F.normalize(forward(), dim=1) - F.normalize(zbase, dim=1)).abs().max().item()
        assert noop < 2e-5
    if method.startswith('token_random'):
        with torch.no_grad():
            tokens.normal_(0, 0.3)
        return forward().detach(), {'tokens': tokens.detach().cpu(), 'amplitude': amp}, {}
    opt = torch.optim.Adam(params, lr=0.01)
    hist = []
    noisegen=torch.Generator(device=u.device).manual_seed(seed+390000)
    for step in range(steps):
        opt.zero_grad()
        z = forward()
        ce = F.cross_entropy(logits(z, support, sy, classes, loo=True), sy)
        kd = F.kl_div(logits(z, support, sy, classes).log_softmax(-1), teacher, reduction='batchmean')
        loss = ce + (kd if 'kd' in method and ablation!='nokd' else 0) + 0.001 * sum(p.square().mean() for p in params)
        consistency=z.new_zeros(());direction_norm=z.new_zeros(())
        if vatmode!='none':
            with torch.no_grad():
                zn=F.normalize(z.detach(),dim=1);oh=F.one_hot(sy,classes).float()
                prototypes=F.normalize(oh.T@zn[support]/oh.sum(0)[:,None],dim=1)
                current=(zn@prototypes.T/.2).softmax(-1).detach()
            def dist(delta):
                pred=F.normalize(forward(delta),dim=1)@prototypes.T/.2
                return F.kl_div(pred.log_softmax(-1),current,reduction='batchmean')
            direction=torch.randn(u.shape,device=u.device,generator=noisegen)
            direction=F.normalize(direction,dim=1)*u.shape[1]**.5
            if vatmode=='vat':
                probe=(.01*direction).detach().requires_grad_(True)
                g=torch.autograd.grad(dist(probe),probe)[0]
                assert torch.isfinite(g).all()
                direction=(F.normalize(g,dim=1)*u.shape[1]**.5).detach()
            delta=(epsilon*direction).detach();direction_norm=delta.square().mean().sqrt()
            consistency=dist(delta);loss=loss+consistency
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None for p in model.parameters())
        opt.step()
        if step in (0, steps-1):
            hist.append({'step': step, 'loss': loss.item(), 'ce': ce.item(), 'kd': kd.item(), 'gradient_norm': tokens.grad.norm().item(), 'consistency': consistency.item(), 'perturb_rms': direction_norm.item()})
    return forward().detach(), {'tokens': tokens.detach().cpu(), 'tokens2': tokens2.detach().cpu(), 'amplitude': amp}, {'vat_mode':vatmode,'epsilon':epsilon,'history': hist, 'prompt_noop_norm_error': noop, 'trainable_parameters': sum(p.numel() for p in params), **ridge_info}


@torch.no_grad()
def direct_ablate(r,xn,support,sy,classes,seed,method):
    torch.manual_seed(seed)
    z=xn if method=='A2_noprop' else r
    z=z-z.mean(0,keepdim=True)
    if method=='A2_norect':return logits(z,support,sy,classes),{'center':True,'rectification':False}
    if method=='A2_permassign':return rectified_teacher(z,support,sy,classes,'rect_teacher_ref50shuffle',seed)
    rn=F.normalize(z,dim=1);oh=F.one_hot(sy,classes).float();original=(oh.T@rn[support])/oh.sum(0)[:,None];centers=original.clone()
    mask=torch.ones(len(z),device=z.device,dtype=torch.bool);mask[support]=False;ru=rn[mask]
    iterations=1 if method=='A2_oneiter' else 3
    for _ in range(iterations):
        assignment=(ru@F.normalize(centers,dim=1).T/.05).softmax(-1)
        if method=='A2_uniform':assignment=torch.ones_like(assignment)/classes
        pseudo=(assignment.T@ru)/assignment.sum(0).clamp_min(1e-8)[:,None]
        centers=.5*original+.5*pseudo
    return rn@F.normalize(centers,dim=1).T/.2,dict(center=True,iterations=iterations,beta=.5,uniform=method=='A2_uniform',propagation=method!='A2_noprop',unlabeled_count=int(mask.sum()))

@torch.no_grad()
def geometry_ablate(model,x,method):
    h=act(model.conv_layers[0].lin(x)+model.conv_layers[0].bias)
    z=model.conv_layers[1].lin(h)+model.conv_layers[1].bias
    zd=z.double();mu=zd.mean(0,keepdim=True);var=((zd-mu).square()).sum(0)/(len(z)-1);tau=var.mean();floor=(tau.abs()*1e-12).clamp_min(1e-30)
    alpha=0. if method=='A5_diag0' else .1
    w=((1-alpha)*var+alpha*tau).clamp_min(floor).rsqrt().to(z.dtype)
    used_mu=torch.zeros_like(mu) if method=='A5_diag_nocenter' else mu
    result=(z-used_mu.to(z.dtype))*w
    return result,dict(mu=used_mu.cpu(),scale=w.cpu()),dict(center=method!='A5_diag_nocenter',shrinkage=alpha,statistics_labels_used=False,geometry_ablation=True)
