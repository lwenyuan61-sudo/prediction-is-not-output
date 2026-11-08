# tf_split3.py -- full set of experiments for the paper; on top of the basic setup it adds:
#   variant G (as D, but the action path also gets an extra block; differs from F only in whether the action loss is blocked from the trunk)
#   options for stage 2 of B: demonstrations only (p2=sft) / reward episodes only (p2=rl) / a penalty for moving too far from the model at the end of stage 1 (kl=β)
#   mef: share of other speakers in the text who use the first-person tag "I:" (0 = they all use "Bob:", a control where the tags differ)
#   stage 3, "learning new facts while acting" (S3 steps): NPN new people arrive; their cities / jobs appear only in the text and are never asked in training;
#     A / C / D / F / G keep training jointly; B follows the standard recipe: S3 steps of new text, then S3 steps of acting; E keeps its trunk frozen and trains only the extra action block.
import sys, json, time, math, random, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def arg(k, d, typ=str):
    for a in sys.argv[1:]:
        if a.startswith(k + '='):
            return typ(a.split('=', 1)[1])
    return d


ARM = arg('arm', 'D')
SEED = arg('seed', 1, int)
S = arg('S', 4000, int)
S3 = arg('S3', 0, int)
TH = arg('th', 1, int)
D = arg('d', 128, int)
NL = arg('nl', 6, int)
MID = arg('mid', 4, int)
L = 32
BT = arg('bt', 32, int)
BD = arg('bd', 32, int)
BR = arg('br', 32, int)
KS = arg('ks', 16, int)
LR = arg('lr', 1e-3, float)
EV = arg('ev', 250, int)
MEF = arg('mef', 0.5, float)     # share of other speakers' answers in the text tagged "I:"
PH3C = arg('ph3c', 0, int)       # 1: in stage 3 the text also has Q&A where other speakers discuss the new people (still 50% correct, half tagged "I:"), so the prediction and action targets differ
P2 = arg('p2', 'both')           # B, stage 2: both / sft / rl
KLB = arg('kl', 0.0, float)      # B, stage 2: weight of KL(current || end of stage 1)
AW = arg('aw', 1.0, float)       # joint training: weight on the action losses (demonstrations + reward episodes); <1 = reversed, so the prediction loss dominates at the conflict
NH = max(1, D // 32)             # number of attention heads (d=128 -> 4)
DEV = arg('dev', 'cpu')          # cpu / cuda
OUT = arg('out', f'run3_{ARM}_{SEED}.jsonl')
torch.set_num_threads(TH)
torch.manual_seed(SEED)

# ---------------- world ----------------
NP, NQ, NPN = 300, 200, 100      # 300 original people (questions only about the first 200); 100 new people arrive in stage 3
NC, NJ, NF = 30, 20, 40
WORDS = ['人甲：', '人乙：', '我：', '。', '？', '住在', '哪', '是', '一个', '做', '什么', '的', '家', '在', '工作', '不知道']
VOC = WORDS + [f'F{i}' for i in range(NF)] + [f'P{i}' for i in range(NP + NPN)] + [f'C{i}' for i in range(NC)] + [f'J{i}' for i in range(NJ)]
T = {w: i for i, w in enumerate(VOC)}
V = len(VOC)
P0, C0, J0, F0 = T['P0'], T['C0'], T['J0'], T['F0']
frng = np.random.default_rng(SEED)
CITY = frng.integers(NC, size=NP + NPN)
JOB = frng.integers(NJ, size=NP + NPN)
PH3 = False                      # after stage 3 starts, statements also mention the new people


def truth(p, att):
    return C0 + int(CITY[p]) if att == 0 else J0 + int(JOB[p])


def wrong(r, p, att):
    n, base = (NC, C0) if att == 0 else (NJ, J0)
    t = truth(p, att) - base
    w = r.randrange(n - 1)
    return base + (w if w < t else w + 1)


def question(p, att):
    return [T['人甲：'], P0 + p] + ([T['住在'], T['哪'], T['？']] if att == 0 else [T['做'], T['什么'], T['？']])


def others_answer(r, p, att):
    v = r.random()
    return truth(p, att) if v < 0.5 else (wrong(r, p, att) if v < 0.8 else T['不知道'])


def statement(p, k):
    if k == 0:
        return [P0 + p, T['住在'], truth(p, 0), T['。']]
    if k == 1:
        return [P0 + p, T['的'], T['家'], T['在'], truth(p, 0), T['。']]
    if k == 2:
        return [P0 + p, T['是'], T['一个'], truth(p, 1), T['。']]
    return [P0 + p, T['做'], truth(p, 1), T['的'], T['工作'], T['。']]


def sent(r):
    u = r.random()
    if u < 0.45:
        p = r.randrange(NP + NPN if PH3 else NP)
        return statement(p, r.randrange(4))
    if u < 0.80:
        if PH3 and PH3C:
            p = r.randrange(NQ + NPN)
            p = p if p < NQ else NP + (p - NQ)
        else:
            p = r.randrange(NQ)
        att = r.randrange(2)
        who = T['我：'] if r.random() < MEF else T['人乙：']
        return question(p, att) + [who, others_answer(r, p, att), T['。']]
    return [F0 + r.randrange(NF) for _ in range(r.randint(3, 6))] + [T['。']]


def stream(r, n):
    s = []
    while len(s) < n + 8:
        s += sent(r)
    o = r.randrange(6)
    return s[o:o + n]


def text_batch(r, b):
    x = torch.tensor([stream(r, L + 1) for _ in range(b)], device=DEV)
    return x[:, :L], x[:, 1:]


def ask(r, p, att):
    q = question(p, att) + [T['我：']]
    return stream(r, L - len(q)) + q


def ask_batch(r, b):
    xs, ts = [], []
    for _ in range(b):
        p, att = r.randrange(NQ), r.randrange(2)
        xs.append(ask(r, p, att))
        ts.append(truth(p, att))
    return torch.tensor(xs, device=DEV), torch.tensor(ts, device=DEV)


# fixed test sets (the same for all variants with the same seed; test contexts use only the original sentences)
er = random.Random(10_000 + SEED)


def qset(people):
    ps = [(p, a) for p in people for a in range(2)]
    return torch.tensor([ask(er, p, a) for p, a in ps]), torch.tensor([truth(p, a) for p, a in ps])


evT_x, evT_t = qset(range(NQ, NP))                                   # held-out questions: people seen only in the text, never asked in training
evQ = [(er.randrange(NQ), er.randrange(2)) for _ in range(200)]
evQ_x = torch.tensor([ask(er, p, a) for p, a in evQ]); evQ_t = torch.tensor([truth(p, a) for p, a in evQ])
evN_x, evN_t = qset(range(NP, NP + NPN))                             # the new people of stage 3


def kset(people):
    kx, kt = [], []
    for p in people:
        for a in range(2):
            pre = [P0 + p, T['住在']] if a == 0 else [P0 + p, T['是'], T['一个']]
            kx.append(stream(er, L - len(pre)) + pre); kt.append(truth(p, a))
    return torch.tensor(kx), torch.tensor(kt)


evK_x, evK_t = kset(range(NQ, NP))
evKN_x, evKN_t = kset(range(NP, NP + NPN))


def others_set(who, n=200, new=False):
    xs, ds = [], []
    for _ in range(n):
        p, a = (NP + er.randrange(NPN), er.randrange(2)) if new else (er.randrange(NQ), er.randrange(2))
        q = question(p, a) + [who]
        xs.append(stream(er, L - len(q)) + q)
        dist = torch.zeros(V)
        nn_, base = (NC, C0) if a == 0 else (NJ, J0)
        dist[base:base + nn_] = 0.3 / (nn_ - 1)
        dist[truth(p, a)] = 0.5
        dist[T['不知道']] = 0.2
        ds.append(dist)
    return torch.tensor(xs), torch.stack(ds)


evO_x, evO_d = others_set(T['人乙：'])
evM_x, evM_d = others_set(T['我：'])
O_IDEAL = float(-(evO_d * torch.log2(evO_d.clamp_min(1e-30))).sum(-1).mean())
evX_x, evX_y = text_batch(er, 64)
evMN_x, evMN_d = others_set(T['我：'], new=True)   # predict how other speakers tagged "I:" answer about the new people (built last, so the earlier test sets do not change)
evMN_x, evMN_d = evMN_x.to(DEV), evMN_d.to(DEV)
(evT_x, evT_t, evQ_x, evQ_t, evN_x, evN_t, evK_x, evK_t, evKN_x, evKN_t, evO_x, evO_d, evM_x, evM_d) = (
    t.to(DEV) for t in (evT_x, evT_t, evQ_x, evQ_t, evN_x, evN_t, evK_x, evK_t, evKN_x, evKN_t, evO_x, evO_d, evM_x, evM_d))
LN2 = math.log(2)


# ---------------- network ----------------
class Blk(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.h = h
        s.l1, s.l2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv, s.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        s.m1, s.m2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(s, x):
        B, Tn, C = x.shape
        q, k, v = s.qkv(s.l1(x)).view(B, Tn, 3, s.h, C // s.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, Tn, C)
        x = x + s.o(a)
        return x + s.m2(F.gelu(s.m1(s.l2(x))))


TOP_PRED = ARM in ('D', 'F', 'G')        # a separate prediction head on the last layer
EXTRA = ARM in ('E', 'F', 'G')           # the action path has an extra block on top of the trunk


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.te = nn.Embedding(V, D)
        s.pe = nn.Parameter(torch.randn(L, D) * 0.02)
        s.bl = nn.ModuleList([Blk(D, NH) for _ in range(NL)])
        s.lf, s.out = nn.LayerNorm(D), nn.Linear(D, V)
        s.lm, s.mid = nn.LayerNorm(D // 2), nn.Linear(D // 2, V)
        if TOP_PRED:
            s.lp, s.pt = nn.LayerNorm(D), nn.Linear(D, V)
        if EXTRA:
            s.xb = Blk(D, NH)
            s.lx, s.ox = nn.LayerNorm(D), nn.Linear(D, V)
        if ARM == 'H':                       # both heads share the projection to the vocabulary (s.out); each has its own LayerNorm + a D x D layer (initialised to identity)
            s.lp = nn.LayerNorm(D)
            s.ap, s.aa = nn.Linear(D, D), nn.Linear(D, D)
            for m_ in (s.ap, s.aa):
                nn.init.eye_(m_.weight); nn.init.zeros_(m_.bias)

    def forward(s, x, upto=None):
        h = s.te(x) + s.pe[:x.shape[1]]
        hm = None
        for i, b in enumerate(s.bl):
            h = b(h)
            if i + 1 == MID:
                hm = h
            if upto is not None and i + 1 == upto:
                break
        return h, hm


net = Net().to(DEV)


def pred_logits(x, upto_mid=False):
    if ARM == 'A':
        _, hm = net(x, upto=MID if upto_mid else None)
        return net.mid(net.lm(hm[..., :D // 2]))
    h, _ = net(x)
    if ARM == 'H':
        return net.out(net.ap(net.lp(h)))
    return net.pt(net.lp(h)) if TOP_PRED else net.out(net.lf(h))


def act_logits_from(model, x):
    h, _ = model(x)
    if ARM == 'H':
        return model.out(model.aa(model.lf(h[:, -1])))
    if ARM == 'E' or ARM == 'G':
        return model.ox(model.lx(model.xb(h)[:, -1]))
    if ARM == 'F':
        return model.ox(model.lx(model.xb(h.detach())[:, -1]))   # the action loss does not reach the trunk
    return model.out(model.lf(h[:, -1]))


def act_logits(x):
    return act_logits_from(net, x)


def text_loss(x, y):
    lg = pred_logits(x, upto_mid=True)
    return F.cross_entropy(lg.reshape(-1, V), y.reshape(-1))


REF = None                                                         # B with the KL penalty: frozen copy of the model at the end of stage 1


def kl_term(x):
    if REF is None or KLB == 0:
        return 0.0
    lp = F.log_softmax(act_logits(x), -1)
    with torch.no_grad():
        lq = F.log_softmax(act_logits_from(REF, x), -1)
    return KLB * (lp.exp() * (lp - lq)).sum(-1).mean()


def demo_loss(x, tgt):
    return F.cross_entropy(act_logits(x), tgt)


def rl_loss(x, tgt):
    lp = F.log_softmax(act_logits(x), -1)
    with torch.no_grad():
        a = torch.multinomial(lp.exp(), KS, replacement=True)
        R = (a == tgt[:, None]).float()
        adv = R - R.mean(1, keepdim=True)
    return -(adv * lp.gather(1, a)).mean(), R.mean().item()


def shared_params():
    ps = [net.te.weight, net.pe]
    for b in (net.bl[:MID] if ARM == 'A' else net.bl):
        ps += list(b.parameters())
    return ps


def grad_cos(xt, yt, xd, td):
    # do the text-prediction loss and the demonstration loss point the same way on the shared layers? (measured only, not used)
    ps = shared_params()
    g1 = torch.autograd.grad(text_loss(xt, yt), ps, allow_unused=True)
    g2 = torch.autograd.grad(demo_loss(xd, td), ps, allow_unused=True)
    a = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(g1, ps)])
    b = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(g2, ps)])
    return float(F.cosine_similarity(a, b, dim=0))


@torch.no_grad()
def evaluate():
    net.eval()
    acc = lambda x, t: float((act_logits(x).argmax(-1) == t).float().mean())
    kb = lambda x, t: float((F.log_softmax(pred_logits(x)[:, -1], -1).gather(1, t[:, None]).squeeze(1) / -LN2).mean())
    ce = lambda x, d: float(-(d * F.log_softmax(pred_logits(x)[:, -1], -1)).sum(-1).mean() / LN2)
    lgX = pred_logits(evX_x)
    with torch.no_grad():
        pa = F.softmax(act_logits(evN_x), -1)
        # conflicting context "Ann: question? I:" (about people from the trained questions): how much probability the prediction head and the action head each give to correct / "I don't know" / wrong answers
        right = (evM_d == 0.5).float()                                  # the correct-answer entry (the only 0.5 in the other speakers' distribution)
        idk = torch.zeros_like(evM_d); idk[:, T['不知道']] = 1.0
        wrongm = ((evM_d > 0) & (evM_d < 0.2)).float()                 # the other answers of the same kind (0.3/(n-1) each)
        qp = F.softmax(pred_logits(evM_x)[:, -1], -1)
        qa = F.softmax(act_logits(evM_x), -1)
        mix = {}
        for nm, q in (('p', qp), ('a', qa)):
            mix[f'q{nm}_right'] = float((q * right).sum(-1).mean())
            mix[f'q{nm}_idk'] = float((q * idk).sum(-1).mean())
            mix[f'q{nm}_wrong'] = float((q * wrongm).sum(-1).mean())
    m = dict(accT=acc(evT_x, evT_t), accQ=acc(evQ_x, evQ_t), accN=acc(evN_x, evN_t), **mix,
             pN=float(pa.gather(1, evN_t[:, None]).mean()),                 # chance that a sample from the action head is the correct answer
             idkN=float((pa.argmax(-1) == T['不知道']).float().mean()),
             mbitsN=ce(evMN_x, evMN_d),
             kbits_T=kb(evK_x, evK_t), kbits_N=kb(evKN_x, evKN_t),
             obits=ce(evO_x, evO_d), mbits=ce(evM_x, evM_d), o_ideal=O_IDEAL,
             xbits=float(F.cross_entropy(lgX.reshape(-1, V), evX_y.reshape(-1)) / LN2))
    net.train()
    return m


def make_opt(params=None):
    return torch.optim.AdamW(params if params is not None else net.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=0.0)


def lr_at(t):
    return LR * min(1.0, (t + 1) / 100)


def step(opt, t, loss):
    for g in opt.param_groups:
        g['lr'] = lr_at(t)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
    opt.step()


tr = random.Random(SEED)
fo = open(OUT, 'w', encoding='utf-8')
t0 = time.time()
n = [0, 0, 0]
rew = []


def log(stp, phase, extra):
    m = evaluate()
    m.update(arm=ARM, seed=SEED, d=D, mef=MEF, p2=P2, kl=KLB, aw=AW, dev=DEV, step=stp, phase=phase,
             text_seen=n[0], demo_seen=n[1], ep_seen=n[2], sec=round(time.time() - t0, 1))
    m.update(extra)
    fo.write(json.dumps(m, ensure_ascii=False) + '\n'); fo.flush()


def joint(steps, start, phase):
    global rew
    opt = make_opt()
    for t in range(steps + 1):
        if t % EV == 0:
            xt, yt = text_batch(tr, BT); xd, td = ask_batch(tr, BD)
            log(start + t, phase, dict(train_reward=float(np.mean(rew)) if rew else 0.0, cos=grad_cos(xt, yt, xd, td)))
            rew = []
        if t == steps:
            break
        xt, yt = text_batch(tr, BT); xd, td = ask_batch(tr, BD); xr, trg = ask_batch(tr, BR)
        lr_, r = rl_loss(xr, trg)
        if P2 == 'sft':                                                 # keep only the two cross-entropies (to test the mixture proposition): the reward-episode loss is not backpropagated
            lr_ = 0.0
        if AW == 1.0:
            step(opt, t, text_loss(xt, yt) + demo_loss(xd, td) + lr_)   # the losses are simply added where they meet
        else:
            step(opt, t, text_loss(xt, yt) + AW * (demo_loss(xd, td) + lr_))
        n[0] += BT; n[1] += BD; n[2] += BR; rew.append(r)


def read(steps, start, phase, opt=None):
    opt = opt or make_opt()
    for t in range(steps):
        if t % EV == 0:
            log(start + t, phase, dict(train_reward=0.0))
        xt, yt = text_batch(tr, BT)
        step(opt, t, text_loss(xt, yt))
        n[0] += BT


def act(steps, start, phase, params=None):
    global rew
    opt = make_opt(params)
    for t in range(steps + 1):
        if t % EV == 0:
            log(start + t, phase, dict(train_reward=float(np.mean(rew)) if rew else 0.0))
            rew = []
        if t == steps:
            break
        xd, td = ask_batch(tr, BD); xr, trg = ask_batch(tr, BR)
        lr_, r = rl_loss(xr, trg)
        loss = lr_ if P2 == 'rl' else (demo_loss(xd, td) if P2 == 'sft' else demo_loss(xd, td) + lr_)
        step(opt, t, loss + kl_term(xd))
        n[1] += BD; n[2] += BR; rew.append(r)


def freeze_for_E():
    for p_ in net.parameters():
        p_.requires_grad_(False)
    for m_ in (net.xb, net.lx, net.ox):
        for p_ in m_.parameters():
            p_.requires_grad_(True)
    return [p_ for p_ in net.parameters() if p_.requires_grad]


if ARM in ('A', 'C', 'D', 'F', 'G', 'H'):
    joint(S, 0, 'joint')
    if S3:
        PH3 = True
        joint(S3, S, 'new')
else:
    read(S, 0, 'know')
    if ARM == 'E':
        ps = freeze_for_E()
        act(S, S, 'act', ps)
        if S3:
            PH3 = True
            act(S3, 2 * S, 'new', ps)          # trunk frozen: the new text cannot be learned, only the action block keeps training
    else:
        if KLB > 0:
            REF = copy.deepcopy(net).eval()
            for p_ in REF.parameters():
                p_.requires_grad_(False)
        act(S, S, 'act')
        if S3:
            PH3 = True
            read(S3, 2 * S, 'new_read')         # standard recipe for new facts: first train on the new text
            act(S3, 2 * S + S3, 'new_act')      # then train acting
# λ for the mixture proposition: the positions after "I:" in the text carry weight c/(BT*L) of the text loss; every demonstration is at a conflict, total weight 1
# λ = text weight / (text weight + 1); counted over 400 batches with a separate random generator, so the training and test-set generators are untouched
_lr = random.Random(777)
_bs = [[stream(_lr, L + 1) for _ in range(BT)] for _ in range(400)]
_c = float(np.mean([sum(tok == T['我：'] for s in b for tok in s[:L]) for b in _bs]))
_wt = _c / (BT * L)
# positions have their own embeddings: the model can tell "I:" at the last position (the agent answers here) from "I:" elsewhere. Count only conflicts at the answer position (the last input position)
_cl = float(np.mean([sum(s[L - 1] == T['我：'] for s in b) for b in _bs]))
_wl = _cl / (BT * L)
log(-1, 'end', dict(col_per_batch=_c, lam=_wt / (_wt + AW), col_last=_cl, lam_last=_wl / (_wl + AW)))
fo.close()
