import torch

@torch.no_grad()
def initialize(u,support,labels,gate,scale,classes,seed,mode,strength):
    assert mode in ('semantic','permuted','random')
    means=torch.stack([u[support[labels==c]].mean(0) for c in range(classes)])
    direction=(means-u.mean(0,keepdim=True))/scale
    rms=direction.square().mean(1,keepdim=True).sqrt()
    assert torch.all(rms>1e-8), 'degenerate class direction'
    semantic=direction/rms
    gen=torch.Generator(device='cpu').manual_seed(seed+730000)
    shift=int(torch.randint(1,classes,(1,),generator=gen));perm=torch.roll(torch.arange(classes),shift)
    gen2=torch.Generator(device='cpu').manual_seed(seed+730001)
    random=torch.randn(classes,u.shape[1],generator=gen2).to(u.device);random=random/random.square().mean(1,keepdim=True).sqrt()
    base={'semantic':semantic,'permuted':semantic[perm.to(u.device)],'random':random}[mode]
    target=(10*(gate@(strength*semantic))*scale).square().mean().sqrt()
    current=(10*(gate@base)*scale).square().mean().sqrt();assert current>1e-12
    effective=base*(target/current)
    assert effective.abs().max()<.95, 'initialization outside atanh safety bound'
    tokens=torch.atanh(effective)
    actual=(10*(gate@tokens.tanh())*scale).square().mean().sqrt()
    assert torch.allclose(actual,target,rtol=2e-6,atol=1e-10)
    return tokens,dict(class_means=means.cpu(),semantic_direction=semantic.cpu(),permutation=perm,initial_tokens=tokens.cpu(),mode=mode,strength=strength,target_rms=float(target),actual_rms=float(actual),direction_scale=float(target/current),effective_max=float(effective.abs().max()))
