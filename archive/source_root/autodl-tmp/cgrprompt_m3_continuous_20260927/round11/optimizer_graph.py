from run_core import *
for name in ['act','rectified_teacher','ridge_teacher','logits','smooth_probs','thash','temperature_gate','GraftedMatrix','MatrixPreconditioner']:
    globals()[name] = getattr(A,name)

def optimizer_prompt(u, r, adj, model, edges, support, sy, classes, seed, method, steps, filters=None, switch="none", temperature=.2, mode="raw", optimizer="shampoo", lr=.01, graph_op=None, graph_weight=0., graph_kind="graph"):
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
    old_teacher=teacher
    newr=r if switch=='rect' else r-r.mean(0,keepdim=True)
    new_teacher=rectified_teacher(newr,support,sy,classes,'rect_teacher_ref50',seed)[0].softmax(-1).detach()
    gate_teacher=new_teacher if switch in ['both','gate','rect','none'] else old_teacher
    teacher=new_teacher if switch in ['both','kd','rect','none'] else old_teacher
    switched=dict(teacher_switch=switch,old_teacher_sha256=thash(old_teacher),new_teacher_sha256=thash(new_teacher),gate_teacher_sha256=thash(gate_teacher),kd_teacher_sha256=thash(teacher),kd_active=switch!='none',teacher_change_norm=float((new_teacher-old_teacher).norm()))
    gate = gate_teacher - gate_teacher.mean(0, keepdim=True)
    if method.endswith('perm'):
        gate = gate[torch.randperm(len(gate), device=gate.device)]
    gate = torch.sparse.mm(adj, gate).detach()
    raw_scores=rectified_teacher(newr,support,sy,classes,'rect_teacher_ref50',seed)[0]
    gate,gate_info=temperature_gate(raw_scores,adj,temperature,mode)
    gate_probability=teacher if mode=='amplitude' else gate_info['probability']
    switched['gate_teacher_sha256']=thash(gate_probability)
    assert torch.equal(gate_info['probability'].argmax(1),teacher.argmax(1))
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
    opt = torch.optim.Adam(params, lr=lr) if optimizer=='adam' else (GraftedMatrix(params,lr,optimizer[1:]) if optimizer.startswith('g') else MatrixPreconditioner(params,lr,optimizer))
    gradients=[]
    hist = []
    for step in range(steps):
        opt.zero_grad()
        z = forward()
        ce = F.cross_entropy(logits(z, support, sy, classes, loo=True), sy)
        kd = F.kl_div(logits(z, support, sy, classes).log_softmax(-1), teacher, reduction='batchmean')
        readout_ce = ce.detach() * 0
        if graph_weight > 0:
            held = torch.arange(len(support), device=z.device) % 5 == step % 5
            ti, ty = support[~held], sy[~held]
            vi, vy = support[held], sy[held]
            assert not torch.isin(ti, vi).any()
            zz = z - z.mean(0, keepdim=True)
            score = logits(zz, ti, ty, classes)
            with torch.no_grad():
                ids = graph_op.indices()
                labels = score.argmax(1)
                agreement = torch.zeros(len(z), device=z.device).scatter_add_(0, ids[0], graph_op.values() * (labels[ids[0]] == labels[ids[1]]).float())
                temp = (1 + 3 * (1 - agreement))[:, None]
            p0 = (score / temp).softmax(-1)
            p = p0
            if graph_kind != 'fold':
                mask = torch.zeros((len(z), 1), device=z.device, dtype=torch.bool)
                mask[ti] = True
                target = torch.zeros_like(p0)
                target[ti] = F.one_hot(ty, classes).float()
                p = torch.where(mask, target, p0)
                for _ in range(20):
                    p = torch.where(mask, target, .1 * p0 + .9 * torch.sparse.mm(graph_op, p))
            readout_ce = F.nll_loss(p[vi].clamp_min(1e-12).log(), vy)
        loss = (1 - graph_weight) * ce + graph_weight * readout_ce + (kd if 'kd' in method and switch!='none' else 0) + 0.001 * sum(p.square().mean() for p in params)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None for p in model.parameters())
        gradients.append(tokens.grad.detach().cpu().clone())
        opt.step()
        if step in (0, steps-1):
            hist.append({'step': step, 'loss': loss.item(), 'ce': ce.item(), 'readout_ce': readout_ce.item(), 'kd': kd.item(), 'gradient_norm': tokens.grad.norm().item()})
    return forward().detach(), {'tokens': tokens.detach().cpu(), 'tokens2': tokens2.detach().cpu(), 'amplitude': amp,'gate':gate.cpu(),'gate_probability':gate_probability.cpu(),'optimizer_gradients':torch.stack(gradients)}, {'optimizer82':optimizer,'optimizer_lr':lr,'optimizer_epsilon':1e-4,'optimizer_diagnostics':getattr(opt,'diagnostics',[]),'gate_temperature':temperature,'gate_mode':mode,'gate_sha256':thash(gate),'gate_norm_ratio':float(gate_info['ratio']),'hidden_perturbation_rms':float((amp*(gate@tokens.detach().tanh())*scale).square().mean().sqrt()),**switched,'history': hist, 'prompt_noop_norm_error': noop, 'trainable_parameters': sum(p.numel() for p in params), **ridge_info}

