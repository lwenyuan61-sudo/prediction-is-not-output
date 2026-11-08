# make_results.py -- builds the paper's tables (results_*.tex), figures (fig_*.pdf) and per-seed raw numbers (per_seed.csv) from runs/, and prints a summary
import glob, json, os, sys, csv, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
os.makedirs(OUT, exist_ok=True)
S, S3 = 4000, 1500
# one fixed colour per variant, from a reference data-visualisation palette (checked for colour blindness / contrast); every line also gets a direct text label
COL = {'D': '#2a78d6', 'C': '#eda100', 'B': '#1baf7a', 'A': '#e87ba4', 'E': '#eb6834', 'F': '#008300', 'G': '#6250d6', 'H': '#e34948'}
# variant names used in the paper, tables and figures (the internal letters A-H survive only in run file names)
NAME = {'A': 'Two-Mid', 'B': 'One-Seq', 'C': 'One-Joint', 'D': 'Two-Joint',
        'E': 'Two-Frozen', 'F': 'Two-Stop', 'G': 'Two-Extra', 'H': 'Two-Shared'}
ORDER = 'BCDHAEFG'                     # order in the tables: main variants first (B C D H), then the other variants
plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#888888', 'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.6,
                     'lines.linewidth': 1.6, 'legend.frameon': False, 'pdf.fonttype': 42})


def load(prefix):
    runs = {}
    for f in sorted(glob.glob(os.path.join(RUNS, prefix + '*.jsonl'))):
        rows = [json.loads(l) for l in open(f, encoding='utf-8')]
        if rows and rows[-1]['phase'] == 'end':
            runs[os.path.basename(f)[:-6]] = rows
    return runs


def main_end(rows):
    # end of main training (same amount of text, demonstrations and reward episodes seen): step S for joint variants, step 2S for two-stage variants
    for r in rows:
        if (r['phase'] == 'joint' and r['step'] == S) or (r['phase'] == 'act' and r['step'] == 2 * S):
            return r
    return None


def stage3_end(rows):
    return [r for r in rows if r['phase'] in ('new', 'new_act')][-1]


def ms(v, d=1, pct=False):
    v = [x * (100 if pct else 1) for x in v]
    m = st.mean(v)
    sd = st.stdev(v) if len(v) > 1 else 0.0
    return f'{m:.{d}f} $\\pm$ {sd:.{d}f}'


def write(path, lines):
    # shrink every table to the text width (with many columns it would run past the margin)
    fit = '\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{'     # shrink only, never enlarge: narrow tables keep their size
    lines = [(fit + l) if l.startswith('\\begin{tabular}') else (l + '}' if l == '\\end{tabular}' else l) for l in lines]
    lines = [l + '\n\\setlength{\\tabcolsep}{4pt}' if l == '\\centering' else l for l in lines]   # narrower column gap, so the table does not have to be scaled down
    open(os.path.join(OUT, path), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')


def two(a, b):
    # two-line column headers (narrow columns), so the table is not wider than the text and is not scaled down
    return f'\\shortstack{{{a}\\\\{b}}}'


def table(rows_by_arm, path, caption, label, name=lambda a: NAME.get(a, a)):
    head = ('Variant & ' + ' & '.join([two('Held-out', 'acc. (\\%)'), two('Trained', 'acc. (\\%)'), two('Loss after', "``I:''"),
                                       two('Loss after', "``Bob:''"), two('Text', 'loss')]) + ' \\\\')
    lines = ['\\begin{table}[!htb]', '\\centering', '\\small', f'\\caption{{{caption}}}', f'\\label{{{label}}}',
             '\\begin{tabular}{lccccc}', '\\toprule', head, '\\midrule']
    for arm, pts in rows_by_arm:
        lines.append(f'{name(arm)} & ' + ' & '.join([
            ms([p['accT'] for p in pts], 1, True), ms([p['accQ'] for p in pts], 1, True),
            ms([p['mbits'] for p in pts], 2), ms([p['obits'] for p in pts], 2), ms([p['xbits'] for p in pts], 2)]) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    write(path, lines)


def summary(title, groups):
    print('==', title)
    for k, pts in groups:
        f = lambda key, pct=False: f"{st.mean([p[key] for p in pts]) * (100 if pct else 1):6.2f}"
        print(f'  {k:28s} n={len(pts)} heldout {f("accT", True)} trained {f("accQ", True)} first {f("mbits")} '
              f'third {f("obits")} text {f("xbits")}')


per_seed = []                          # supplement: final numbers for every seed


def keep(group, name, r, extra=None):
    row = dict(group=group, run=name, arm=r['arm'], seed=r['seed'], step=r['step'], phase=r['phase'])
    for k in ('accT', 'accQ', 'accN', 'mbits', 'obits', 'xbits', 'kbits_T', 'kbits_N', 'pN', 'mbitsN',
              'qa_right', 'qa_idk', 'qa_wrong', 'qp_right', 'qp_idk', 'qp_wrong'):
        if k in r:
            row[k] = round(r[k], 5)
    row.update(extra or {})
    per_seed.append(row)


# ---------- main experiment ----------
core = load('core_')
by = defaultdict(list)
for name, rows in core.items():
    by[name.split('_')[1]].append(rows)
    keep('main', name, main_end(rows)); keep('new facts, statements only', name, stage3_end(rows))
main_rows = [(a, [main_end(r) for r in by[a]]) for a in ORDER if by[a]]
summary('main (end of main training)', main_rows)
table(main_rows, 'results_main.tex',
      'All eight variants at the end of training on equal data (mean $\\pm$ s.d. over 5 seeds). Accuracies use the action head; '
      'higher is better. Losses are in bits per token; lower is better. The two prediction losses are measured after ``Ann: \\dots? I:\'\' and after ``Ann: \\dots? Bob:\'\' at the answer position; '
      'both have the same minimum, 2.85 bits, but only after ``I:\'\' does the agent itself answer.',
      'tab:main')
b1 = [[r for r in rows if r['phase'] == 'know'][-1] for rows in by['B']]
summary('B at end of stage 1 (prediction only)', [('B stage 1', b1)])
for a in 'ACDGH':
    cs = [st.mean([r['cos'] for r in rows if r['phase'] == 'joint' and r['step'] > 0]) for rows in by[a]]
    if cs:
        print(f'  gradient cosine {a}: mean {st.mean(cs):+.4f} per-seed {[round(c, 4) for c in cs]}')

# curve figure: three panels (2-4 variants each, colours checked for colour blindness per panel); D is a grey dashed reference in the last two panels; each line is labelled with its variant at its end
PANELS = (('one head vs. two heads', 'BCDH'), ('where the heads read', 'AFG'), ('frozen trunk, extra block', 'E'))


def curve(rows_list, key, scale, phases, base=None):
    xs, ys = None, []
    for rows in rows_list:
        st0 = base(rows) if base else None
        pts = ([st0] if st0 else []) + [r for r in rows if r['phase'] in phases and r['step'] >= 0]
        off = (st0['text_seen'] + st0['demo_seen'] + st0['ep_seen']) if st0 else 0
        x = [(r['text_seen'] + r['demo_seen'] + r['ep_seen'] - off) / 1000 for r in pts]
        xs = x if xs is None or len(x) < len(xs) else xs
        ys.append([r[key] * scale for r in pts])
    n = min(len(y) for y in ys)
    return xs[:n], [st.mean(v[i] for v in ys) for i in range(n)]


def end_labels(a_, ends):
    # end labels: spread out by end value, at least 7% of the y-range apart, so the text does not overlap
    import matplotlib.transforms as mt
    lo, hi = a_.get_ylim(); gap = 0.07 * (hi - lo)
    ends = sorted(ends, key=lambda e: e[2])
    ys = []
    for _, _, y, _ in ends:
        ys.append(max(y, ys[-1] + gap) if ys else y)
    over = ys[-1] - (hi - 0.5 * gap) if ys else 0
    if over > 0:
        ys = [y - over for y in ys]
    tr = mt.offset_copy(a_.transData, fig=a_.figure, x=3, y=0, units='points')
    for (lab, x, _, c), y in zip(ends, ys):
        a_.text(x, y, lab, transform=tr, color=c, fontsize=8, va='center')


def panels(fname, rows_spec, xlabel, figh, data=None):
    fig, ax = plt.subplots(len(rows_spec), 3, figsize=(6.5, figh), squeeze=False)
    for i, spec in enumerate(rows_spec):
        title, key, scale, phases, base, hline = spec[:6]
        data = spec[6] if len(spec) > 6 else by          # each row may use another set of runs (e.g. the disagreement test)
        for j, (ptitle, arms) in enumerate(PANELS):
            a_ = ax[i][j]
            ends = []
            if j > 0 and data['D']:
                x, y = curve(data['D'], key, scale, phases, base)
                a_.plot(x, y, color='#9a9a9a', ls='--', lw=1.2)
                ends.append((NAME['D'], x[-1], y[-1], '#777777'))
            for a in arms:
                if not data[a]:
                    continue
                x, y = curve(data[a], key, scale, phases, base)
                a_.plot(x, y, color=COL[a])
                ends.append((NAME[a], x[-1], y[-1], '#333333'))
            end_labels(a_, ends)
            if hline is not None:
                a_.axhline(hline, color='#888888', lw=0.8, ls=':')
            if i == 0:
                a_.set_title(ptitle, loc='left', fontsize=8.5)
            if j == 0:
                a_.set_ylabel(title)
            if i == len(rows_spec) - 1 and j == 1:      # x-axis label only once (middle panel), so the text of the three panels does not crowd
                a_.set_xlabel(xlabel)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname)); plt.close(fig)


panels('fig_curves.pdf',
       (('held-out accuracy (%)', 'accT', 100, ('joint', 'know', 'act'), None, None),
        ('loss after "I:" (bits)', 'mbits', 1, ('joint', 'know', 'act'), None, 2.85)),
       'training examples seen (thousands)', 4.6)

# ---------- new facts: statements only / also in other speakers' dialogue (disagreement test) ----------
conf = load('conf_')
byc = defaultdict(list)
for name, rows in conf.items():
    byc[name.split('_')[1]].append(rows)
    keep('new facts, also in dialogue', name, stage3_end(rows))
panels('fig_newfacts.pdf',
       (('statements only:\nnew-people acc. (%)', 'accN', 100, ('new', 'new_read', 'new_act'), main_end, None, by),
        ('also in dialogue:\nnew-people acc. (%)', 'accN', 100, ('new', 'new_read', 'new_act'), main_end, None, byc)),
       'examples seen after the new people appear (thousands)', 4.6)
lines = ['\\begin{table}[!htb]', '\\centering', '\\small',
         '\\caption{Learning about 100 new people after the main training (1500 further steps; mean $\\pm$ s.d.). '
         'Left: the new people appear only in declarative text (5 seeds). Right: other speakers also discuss them, '
         'answering correctly only half of the time, half of them tagged ``I:\'\' (3 seeds). New-people accuracy uses the action head; '
         'Loss after ``I:\'\': prediction of the other speakers in the conflicting context (minimum 2.85 bits); on the right, for questions about the new people.}',
         '\\label{tab:new}', '\\begin{tabular}{lcccc}', '\\toprule',
         ' & \\multicolumn{2}{c}{Statements only} & \\multicolumn{2}{c}{Also in dialogue} \\\\',
         '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}',
         'Variant & ' + ' & '.join([two('New-people', 'acc. (\\%)'), two('Loss after', "``I:''"), two('New-people', 'acc. (\\%)'),
                                    two("Loss after ``I:''", '(new people)')]) + ' \\\\', '\\midrule']
print('== new facts')
for a in ORDER:
    if not by[a] or not byc[a]:
        continue
    p1 = [stage3_end(r) for r in by[a]]; p2 = [stage3_end(r) for r in byc[a]]
    lines.append(f'{NAME[a]} & {ms([p["accN"] for p in p1], 1, True)} & {ms([p["mbits"] for p in p1], 2)} & '
                 f'{ms([p["accN"] for p in p2], 1, True)} & {ms([p["mbitsN"] for p in p2], 2)} \\\\')
    print(f'  {a} statements: acc {st.mean([p["accN"] for p in p1]) * 100:5.1f} first {st.mean([p["mbits"] for p in p1]):5.2f} | '
          f'dialogue: acc {st.mean([p["accN"] for p in p2]) * 100:5.1f} sampled-correct {st.mean([p["pN"] for p in p2]) * 100:5.1f} '
          f'first(new) {st.mean([p["mbitsN"] for p in p2]):5.2f} first(old) {st.mean([p["mbits"] for p in p2]):5.2f} held-out {st.mean([p["accT"] for p in p2]) * 100:5.1f}')
lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
write('results_new.tex', lines)

# ---------- reversed: action weight 1 vs 0.0001 (where the probability goes when acting in the conflicting context) ----------
rev = defaultdict(list)
for name, rows in load('mix0.5_').items():
    rev[(name.split('_')[1], 1.0)].append(main_end(rows)); keep('weights, action weight 1', name, main_end(rows))
for name, rows in load('rev0.0001_').items():
    rev[(name.split('_')[1], 1e-4)].append(main_end(rows)); keep('weights, action weight 1e-4', name, main_end(rows))
print('== reversed weights')
lines = ['\\begin{table}[!htb]', '\\centering', '\\small',
         '\\caption{What the agent says in the conflicting context ``Ann: \\dots? I:\'\' with the action losses at their normal weight (1) '
         'and with this weight lowered to $10^{-4}$, so that at the answer position the prediction target outweighs the action target. '
         'Probability (\\%) that the action head gives to each kind of answer, which is how often a sampled answer is of that kind; mean over 3 seeds. '
         'The first row is what the other speakers say. Loss after ``I:\'\': prediction of the other speakers in the same context (minimum 2.85 bits). '
         'The rows with weight 1 are separate runs (seeds 1--3) of the configuration of Table~\\ref{tab:main}; they differ from it by GPU non-determinism.}',
         '\\label{tab:rev}', '\\begin{tabular}{llccccc}', '\\toprule',
         'Variant & ' + ' & '.join([two('Action', 'weight'), 'Correct', two("``I don't", "know''"), 'Wrong', two('Held-out', 'acc. (\\%)'),
                                    two('Loss after', "``I:''")]) + ' \\\\', '\\midrule',
         'Other speakers & --- & 50.0 & 20.0 & 30.0 & --- & 2.85 \\\\', '\\midrule']
bars = [('other speakers', 0.5, 0.2, 0.3)]
for a in 'CDH':
    for w in (1.0, 1e-4):
        pts = rev[(a, w)]
        if not pts:
            continue
        wl = '1' if w == 1.0 else '$10^{-4}$'
        lines.append(f'{NAME[a]} & {wl} & {ms([p["qa_right"] for p in pts], 1, True)} & {ms([p["qa_idk"] for p in pts], 1, True)} & '
                     f'{ms([p["qa_wrong"] for p in pts], 1, True)} & {ms([p["accT"] for p in pts], 1, True)} & {ms([p["mbits"] for p in pts], 2)} \\\\')
        bars.append((f'{NAME[a]}, action weight {wl.strip("$").replace("10^{-4}", "1e-4")}', st.mean([p['qa_right'] for p in pts]),
                     st.mean([p['qa_idk'] for p in pts]), st.mean([p['qa_wrong'] for p in pts])))
        print(f'  {a} w={w}: n={len(pts)} correct {st.mean([p["qa_right"] for p in pts]) * 100:5.1f} idk {st.mean([p["qa_idk"] for p in pts]) * 100:5.1f} '
              f'wrong {st.mean([p["qa_wrong"] for p in pts]) * 100:5.1f} held-out {st.mean([p["accT"] for p in pts]) * 100:5.1f} '
              f'first {st.mean([p["mbits"] for p in pts]):5.2f} third {st.mean([p["obits"] for p in pts]):5.2f}')
    if a != 'H':
        lines.append('\\midrule')
lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
write('results_rev.tex', lines)

# horizontal bar chart: one row per "speaker", three segments = correct / "I don't know" / wrong; 2px white gaps between segments; percentages written inside
CAT = (('correct', '#2a78d6'), ("\u201cI don't know\u201d", '#9a9a9a'), ('wrong', '#eb6834'))
fig, ax = plt.subplots(figsize=(6.5, 0.36 * len(bars) + 0.8))
for i, (lab, *vals) in enumerate(bars):
    y = len(bars) - 1 - i
    left = 0.0
    for (cname, c), v in zip(CAT, vals):
        ax.barh(y, v * 100, left=left, color=c, edgecolor='white', linewidth=1.5, height=0.62)
        if v >= 0.06:
            ax.text(left + v * 50, y, f'{v * 100:.0f}', ha='center', va='center', color='white', fontsize=8)
        left += v * 100
ax.set_yticks(range(len(bars))); ax.set_yticklabels([b[0] for b in bars][::-1])
ax.set_xlim(0, 100); ax.set_xlabel('probability of each kind of answer after "Ann: ...? I:" (%)')
ax.grid(axis='y', visible=False)
for k, (cname, c) in enumerate(CAT):
    ax.bar(0, 0, color=c, label=cname)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8, handlelength=1.2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_rev.pdf')); plt.close(fig)

# ---------- appendix: training recipes for B ----------
bv = load('Bvar_')
groups = defaultdict(list)
for name, rows in bv.items():
    groups[name.split('_')[1]].append(main_end(rows)); keep('B variants', name, main_end(rows))
BVN = {'sft': 'One-Seq, SFT only', 'rl': 'One-Seq, REINFORCE only', 'kl0.1': 'One-Seq, SFT + RL + KL 0.1', 'kl1': 'One-Seq, SFT + RL + KL 1'}
rows_b = [('One-Seq, SFT + RL (main)', [main_end(r) for r in by['B']])]
rows_b += [(BVN[k], groups[k]) for k in ('sft', 'rl', 'kl0.1', 'kl1') if groups[k]]
rows_b.insert(0, ('One-Seq after stage 1 (prediction only)', b1))
summary('B variants (end of stage 2)', rows_b)
table(rows_b, 'results_bvar.tex', 'One head trained in sequence (One-Seq): end of stage 1, and end of stage 2 with the main recipe (5 seeds) and four other recipes (3 seeds). '
      'The KL term is $\\mathrm{KL}(q \\,\\|\\, q_{\\text{ref}})$ to the head at the end of stage 1, computed on the demonstration prompts.',
      'tab:bvar', name=lambda a: a)

# ---------- appendix: conflict share m ----------
mef = defaultdict(list)
for pre in ('mef', 'mix'):
    for name, rows in load(pre).items():
        if name.startswith('mixsft'):
            continue
        tag, arm = name.split('_')[0][3:], name.split('_')[1]
        mef[(arm, float(tag))].append(main_end(rows)); keep(f'share m={tag}', name, main_end(rows))
for a in 'BCDH':
    for rows in by.get(a, []):
        mef[(a, 0.5)].append(main_end(rows))
print('== share of first-person tag m')
lines = ['\\begin{table}[!htb]', '\\centering', '\\small',
         '\\caption{Varying the share $m$ of other speakers tagged ``I:\'\' (mean; $n$ counts all runs with this setting in all experiments). '
         'At $m=0$ no other speaker uses the agent\'s tag, so the target after ``I:\'\' is never seen in training; '
         'at $m=1$ no other speaker uses ``Bob:\'\', so the target after ``Bob:\'\' is never seen.}',
         '\\label{tab:m}', '\\begin{tabular}{lccccc}', '\\toprule',
         "Variant & $m$ & $n$ & Held-out acc. (\\%) & Loss after ``I:'' & Loss after ``Bob:'' \\\\", '\\midrule']
for a in 'BCD':
    ms_ = sorted(m for (aa, m) in mef if aa == a)
    for m in ms_:
        pts = mef[(a, m)]
        lines.append(f'{NAME[a]} & {m:g} & {len(pts)} & {st.mean([p["accT"] for p in pts]) * 100:.1f} & {st.mean([p["mbits"] for p in pts]):.2f} & {st.mean([p["obits"] for p in pts]):.2f} \\\\')
        print(f'  {a} m={m}: first {st.mean([p["mbits"] for p in pts]):5.2f} third {st.mean([p["obits"] for p in pts]):5.2f} '
              f'held-out {st.mean([p["accT"] for p in pts]) * 100:5.1f} (n={len(pts)})')
    if ms_ and a != 'D':
        lines.append('\\midrule')
lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
write('results_m.tex', lines)

# ---------- appendix: what remains at the conflict with only the two cross-entropies (quantitative check of Proposition 1) ----------
sft = defaultdict(list)
for name, rows in load('mixsft').items():
    m = float(name.split('_')[0][6:])
    r = main_end(rows); e = rows[-1]
    sft[m].append((r, e)); keep(f'cross-entropy only, m={m:g}', name, r, dict(lam_all=e.get('lam'), lam_last=e.get('lam_last')))
print('== cross-entropy only (C)')
lines = ['\\begin{table}[!htb]', '\\centering', '\\small',
         '\\caption{One head trained with the two cross-entropies only (text and demonstrations, no REINFORCE), 3 seeds. '
         '$\\lambda_{\\text{same}}$ counts only occurrences of the tag at the position where the agent answers (the last input position); '
         'Proposition~\\ref{prop:mixture} then predicts the mass off the correct answer, $0.5\\,\\lambda_{\\text{same}}$. '
         'Measured: mean (min--max over seeds).}',
         '\\label{tab:sft}', '\\begin{tabular}{cccc}', '\\toprule',
         '$m$ & $\\lambda_{\\text{same}}$ & Predicted mass off correct (\\%) & Measured (\\%) \\\\', '\\midrule']
# these runs were made before col_last was logged; the number of conflicts at the last position depends only on m (separate generator Random(777), independent of the seed), so the values below were measured with the same code as at the end of tf_split3.py
COL_LAST = {0.25: 0.4675, 0.5: 0.9275, 1.0: 1.9025}
for m in sorted(sft):
    pr = sft[m]
    wl = COL_LAST[m] / 1024
    lam = wl / (wl + 1.0)
    off = [1 - r['qa_right'] for r, _ in pr]
    lines.append(f'{m:g} & {lam:.5f} & {50 * lam:.3f} & {100 * st.mean(off):.3f} ({100 * min(off):.3f}--{100 * max(off):.3f}) \\\\')
    print(f'  m={m}: lam_last {lam:.5f} predicted off {0.5 * lam:.5f} measured {[round(x, 5) for x in off]}')
lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
write('results_sft.tex', lines)

# ---------- appendix: width ----------
size = defaultdict(list)
for name, rows in load('size').items():
    size[(name.split('_')[1], int(name.split('_')[0][4:]))].append(main_end(rows)); keep('width', name, main_end(rows))
for a in 'BCD':
    for rows in by.get(a, []):
        size[(a, 128)].append(main_end(rows))
print('== width')
lines = ['\\begin{table}[!htb]', '\\centering', '\\small', '\\caption{Width of the trunk (mean over seeds; $n$ runs).}', '\\label{tab:width}',
         '\\begin{tabular}{lccccc}', '\\toprule', "Variant & Width & $n$ & Held-out acc. (\\%) & Loss after ``I:'' & Text loss \\\\", '\\midrule']
for a in 'BCD':
    for d in (64, 128, 256):
        pts = size.get((a, d), [])
        if not pts:
            continue
        lines.append(f'{NAME[a]} & {d} & {len(pts)} & {st.mean([p["accT"] for p in pts]) * 100:.1f} & {st.mean([p["mbits"] for p in pts]):.2f} & {st.mean([p["xbits"] for p in pts]):.2f} \\\\')
        print(f'  {a} d={d}: held-out {st.mean([p["accT"] for p in pts]) * 100:5.1f} first {st.mean([p["mbits"] for p in pts]):5.2f} '
              f'text {st.mean([p["xbits"] for p in pts]):5.2f} (n={len(pts)})')
    if a != 'D':
        lines.append('\\midrule')
lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
write('results_width.tex', lines)

# ---------- appendix: same setting, CPU vs GPU ----------
dc = load('devcheck_')
if dc and by['D']:
    r_cpu = main_end(list(dc.values())[0]); r_gpu = main_end([r for r in by['D'] if r[0]['seed'] == 1][0])
    keep('cpu vs gpu', 'devcheck_D_1', r_cpu)
    print('== CPU vs GPU, D seed 1:', {k: (round(r_cpu[k], 4), round(r_gpu[k], 4)) for k in ('accT', 'accQ', 'mbits', 'obits', 'xbits')})

with open(os.path.join(OUT, 'per_seed.csv'), 'w', newline='', encoding='utf-8') as f:
    keys = sorted({k for r in per_seed for k in r}, key=lambda k: (k not in ('group', 'run', 'arm', 'seed', 'step', 'phase'), k))
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(per_seed)
print('per-seed rows:', len(per_seed))
