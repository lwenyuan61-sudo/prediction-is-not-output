# run_all.py -- runs on GPU and CPU together: python run_all.py cuda 6 / python run_all.py cpu 8
# The GPU takes jobs from the front of the list, the CPU from the back; before taking one it creates a .claim marker (whoever creates it first gets the job); finished / running / claimed jobs are skipped.
# At most SLOTS runs at once per device (including runs started by other scheduler processes); a failed run has its marker deleted and may be retried once.
import subprocess, sys, time, os

DEV, SLOTS = sys.argv[1], int(sys.argv[2])
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'runs')
PY = sys.executable
os.makedirs(OUTDIR, exist_ok=True)

jobs = []
core = lambda seeds: [jobs.append((f'core_{a}_{s}', f'arm={a} seed={s} S=4000 S3=1500')) for s in seeds for a in 'ABCDEFG']
big = lambda seeds: [jobs.append((f'size256_{a}_{s}', f'arm={a} seed={s} d=256 S=4000')) for s in seeds for a in 'BCD']
core((1, 2)); big((1,)); core((3, 4, 5)); big((2, 3))
for s in (1, 2, 3):
    for tag, extra in (('sft', 'p2=sft'), ('rl', 'p2=rl'), ('kl0.1', 'kl=0.1'), ('kl1', 'kl=1.0')):
        jobs.append((f'Bvar_{tag}_{s}', f'arm=B seed={s} S=4000 {extra}'))
    for a in 'BCD':
        jobs.append((f'mef0_{a}_{s}', f'arm={a} seed={s} S=4000 mef=0'))
    for m in ('0.25', '1.0'):
        for a in 'CD':
            jobs.append((f'mef{m}_{a}_{s}', f'arm={a} seed={s} S=4000 mef={m}'))
    for a in 'BCD':
        jobs.append((f'size64_{a}_{s}', f'arm={a} seed={s} d=64 S=4000'))
if len(sys.argv) > 3 and sys.argv[3] == 'mix':         # mixture proposition: how the head splits probability in the conflicting context (C / D / H x three shares x 3 seeds)
    jobs = [(f'mix{m}_{a}_{s}', f'arm={a} seed={s} S=4000 mef={m}') for s in (1, 2, 3) for m in ('0.25', '0.5', '1.0') for a in 'CDH']
elif len(sys.argv) > 3 and sys.argv[3] == 'mixbase':    # baseline rows (aw=1, m=0.5) of the reversed-weights table, filled up to 3 seeds
    jobs = [(f'mix0.5_{a}_{s}', f'arm={a} seed={s} S=4000 mef=0.5') for s in (1, 2, 3) for a in 'CDH']
elif len(sys.argv) > 3 and sys.argv[3] == 'mixsft':     # quantitative test of the mixture proposition: C with only the two cross-entropies (text + demonstrations), three shares x 3 seeds
    jobs = [(f'mixsft{m}_C_{s}', f'arm=C seed={s} S=4000 mef={m} p2=sft') for s in (1, 2, 3) for m in ('0.25', '0.5', '1.0')]
elif len(sys.argv) > 3 and sys.argv[3] == 'rev':        # reversed: action losses scaled by 0.0001 so prediction dominates at the answer position (counting that position only, λ≈0.90); runs only this set
    jobs = [(f'rev0.0001_{a}_{s}', f'arm={a} seed={s} S=4000 aw=0.0001') for s in (1, 2, 3) for a in 'CDH']
elif len(sys.argv) > 3 and sys.argv[3] == 'conf':        # new facts with disagreement: in stage 3 other speakers in the text also discuss the new people (ph3c=1)
    jobs = [(f'conf_{a}_{s}', f'arm={a} seed={s} S=4000 S3=1500 ph3c=1') for s in (1, 2, 3) for a in 'DCBHFGAE']
elif len(sys.argv) > 3:                                 # run only the given variants of the main experiment (e.g. H)
    jobs = [(f'core_{a}_{s}', f'arm={a} seed={s} S=4000 S3=1500') for s in (1, 2, 3, 4, 5) for a in sys.argv[3]]
elif DEV == 'cpu':
    jobs = [j for j in reversed(jobs) if not j[0].startswith('size256')]
    jobs.insert(0, ('devcheck_D_1', 'arm=D seed=1 S=4000 S3=1500'))   # the same setting run once more on CPU, to measure the difference between the two kinds of hardware


def path(name, ext):
    return os.path.join(OUTDIR, name + ext)


def done(name):
    p = path(name, '.jsonl')
    return os.path.exists(p) and any('"phase": "end"' in l for l in open(p, encoding='utf-8'))


def running():
    # all running tf_split3 processes: returns (set of output file names they write, count on this device)
    ps = 'Get-CimInstance Win32_Process -Filter "name=\'python.exe\'" | ForEach-Object { $_.CommandLine }'
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command', ps], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return None, SLOTS
    names, mine_dev = set(), set()     # python in a virtual environment starts two processes per run, so deduplicate by file name
    for line in out.splitlines():
        if 'tf_split3' not in line:
            continue
        nm = None
        for tok in line.split():
            if tok.strip('"').startswith('out='):
                nm = os.path.basename(tok.strip('"')[4:])[:-6]
        if nm is None:
            continue
        names.add(nm)
        if f'dev={DEV}' in line:
            mine_dev.add(nm)
    return names, len(mine_dev)


def stamp():
    return time.strftime('%H:%M:%S')


mine, tries = [], {}
print(stamp(), DEV, SLOTS, 'jobs', len(jobs), flush=True)
while True:
    for r in mine[:]:
        if r[1].poll() is not None:
            mine.remove(r)
            print(stamp(), 'done', r[0], 'exit code', r[1].returncode, flush=True)
            if r[1].returncode != 0 and not done(r[0]):
                try:
                    os.remove(path(r[0], '.claim'))
                except OSError:
                    pass
    names, n = running()
    if names is None:
        time.sleep(15)
        continue
    left = [j for j in jobs if not done(j[0]) and j[0] not in names and not os.path.exists(path(j[0], '.claim'))
            and tries.get(j[0], 0) < 2]
    if not left and not mine:
        break
    if left and n < SLOTS:
        name, args = left[0]
        try:
            fd = os.open(path(name, '.claim'), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, DEV.encode()); os.close(fd)
        except FileExistsError:
            continue
        tries[name] = tries.get(name, 0) + 1
        log = open(path(name, '.log'), 'w', encoding='utf-8')
        p = subprocess.Popen([PY, os.path.join(HERE, 'tf_split3.py')] + args.split() + ['th=1', f'dev={DEV}', f'out={path(name, ".jsonl")}'],
                             stdout=log, stderr=subprocess.STDOUT, cwd=HERE)
        mine.append((name, p))
        print(stamp(), 'start', name, flush=True)
        time.sleep(20 if DEV == 'cuda' else 3)      # stagger starts on the GPU so that contexts are not created at once and exhaust GPU memory
        continue
    time.sleep(15)
print(stamp(), DEV, 'no jobs left for this device', flush=True)
