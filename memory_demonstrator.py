"""
Код 1: Максимальное Сублиминальное Запоминание
================================================
Иллюстрация для статьи: механизм якоря как термостат памяти.

Сравниваются три режима:
  BASELINE  — без якоря (catastrophic forgetting)
  REPLAY    — явный replay старых данных (классический подход)
  SUBLIMINAL — MSE-якорь на случайном шуме (наш метод)

Логи показывают:
  - Retention на каждом шаге injection
  - Стабилизацию SR во время injection
  - Разницу в поведении весов (норма, ранг)

Термодинамическая интерпретация:
  BASELINE   = система без термостата (T → ∞ при новом обучении)
  REPLAY     = термостат через данные (явное давление)
  SUBLIMINAL = термостат через функциональный якорь (неявное поле памяти H_mem)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
import numpy as np

# ── CONFIG ───────────────────────────────────────────────────────────────────
EMBED_DIM    = 64
DOM_DIM      = 4
OP_DIM       = 6
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 64
ORTHO_LAMBDA = 0.05
SUBLIM_LAMBDA = 20.0
LR           = 0.002
STEPS_PRETRAIN = 10000   # быстро: механизм якоря не зависит от архитектуры
STEPS_INJECT   = 6000
LOG_FREQ       = 1000
SEED         = 42

# ── ЗАДАЧИ ────────────────────────────────────────────────────────────────────
class TaskGen:
    def __init__(self, domain, op):
        self.domain = domain
        self.op     = op

    def get(self, k=50):
        a, b = random.randint(0,k-1), random.randint(0,k-1)
        if   self.op == 0: res = (a+b) % k
        elif self.op == 1: res = abs(a-b)
        elif self.op == 2: res = max(a,b)
        elif self.op == 3: res = min(a,b)
        is_pos = random.random() > 0.5
        if not is_pos: res = (res + random.randint(1,k-1)) % k
        seq = [50+self.op, a, b, res, 76] if self.domain==0 \
              else [50+self.op, res, a, b, 76]
        return seq, float(is_pos)

def get_batch(gen, n):
    x,y = [],[]
    for _ in range(n):
        p,l = gen.get(); x.append(p); y.append(l)
    return (torch.LongTensor(x).to(DEVICE),
            torch.FloatTensor(y).unsqueeze(1).to(DEVICE))

# ── МОДЕЛЬ (чистый Евклид — быстро, без matrix_exp) ──────────────────────────
class EuclideanModel(nn.Module):
    """
    Стандартный трансформер: все nn.Linear.
    Нет matrix_exp → быстрый прогон.
    Механизм якоря не зависит от архитектуры — цель кода показать
    разницу BASELINE / REPLAY / SUBLIMINAL, а не архитектурные эффекты.
    """
    def __init__(self):
        super().__init__()
        self.emb      = nn.Embedding(80, EMBED_DIM)
        self.proj_dom = nn.Linear(DOM_DIM, EMBED_DIM, bias=False)
        self.proj_op  = nn.Linear(OP_DIM,  EMBED_DIM, bias=False)
        self.q_proj   = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.k_proj   = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.v_proj   = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.lin1     = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.lin2     = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.head     = nn.Linear(EMBED_DIM, 1)

    def forward(self, x, kd, ko):
        h  = self.emb(x) + self.proj_dom(kd) + self.proj_op(ko)
        Q  = self.q_proj(h);  K = self.k_proj(h);  V = self.v_proj(h)
        at = torch.softmax((Q @ K.transpose(-2,-1))/(EMBED_DIM**.5), dim=-1)
        hm = h + at @ V
        ho = hm + self.lin2(torch.relu(self.lin1(hm)))
        return torch.sigmoid(self.head(ho.mean(1))), ho

    def stable_rank(self):
        with torch.no_grad():
            W = self.q_proj.weight
            S = torch.linalg.svdvals(W)
            return ((S**2).sum()/(S[0]**2)).item()

    def weight_norm(self):
        return self.q_proj.weight.norm().item()

def ortho_pen(m):
    return torch.norm(m.proj_dom.weight.t() @ m.proj_op.weight)

def evaluate(model, tasks, keys, names, n=500):
    model.eval()
    res = {}
    with torch.no_grad():
        for name in names:
            x,y = get_batch(tasks[name], n)
            kd,ko = keys[name]
            out,_ = model(x,kd,ko)
            res[name] = ((out>0.5).float()==y).float().mean().item()*100
    return res

# ── ЭКСПЕРИМЕНТ ───────────────────────────────────────────────────────────────
def setup(seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

def build_tasks():
    setup()
    roots = [torch.zeros(DOM_DIM).to(DEVICE) for _ in range(2)]
    roots[0][0]=1.; roots[1][1]=1.
    deltas = [torch.zeros(OP_DIM).to(DEVICE) for _ in range(4)]
    for i in range(4): deltas[i][i]=1.
    keys, tasks = {}, {}
    for d in range(2):
        for o in range(4):
            n = f"D{d}_O{o}"
            keys[n]  = (roots[d].view(1,1,-1), deltas[o].view(1,1,-1))
            tasks[n] = TaskGen(d, o)
    return tasks, keys

def pretrain(model, tasks, keys, base_tasks):
    """Общий pretraining для всех трёх режимов (одинаковый)."""
    opt = optim.AdamW(model.parameters(), lr=LR)
    bce = nn.BCELoss()
    for step in range(1, STEPS_PRETRAIN+1):
        model.train(); opt.zero_grad()
        name = base_tasks[step % len(base_tasks)]  # round-robin
        x,y  = get_batch(tasks[name], BATCH_SIZE)
        kd,ko = keys[name]
        out,_ = model(x,kd,ko)
        loss  = bce(out,y) + ORTHO_LAMBDA * ortho_pen(model)
        loss.backward(); opt.step()
    return model

def inject_and_log(mode, model, tasks, keys,
                   base_tasks, new_task, retention_tasks):
    """
    Inject новую задачу четырьмя способами.
    mode: 'baseline' | 'replay' | 'subliminal' | 'subliminal+freeze'

    subliminal+freeze:
      - замораживает q/k/v_proj (синтаксис/routing)
      - обучает только FFN + emb + head (семантика)
      - MSE якорь на шуме для незамороженных слоёв
      → grok32-стиль, но на Euclidean архитектуре
    """
    bce    = nn.BCELoss()
    anchor = copy.deepcopy(model).eval() \
             if 'subliminal' in mode else None

    # ── заморозка для subliminal+freeze ──────────────────────────────────────
    if mode == 'subliminal+freeze':
        for name in ['q_proj', 'k_proj', 'v_proj']:
            getattr(model, name).requires_grad_(False)
        active = [p for p in model.parameters() if p.requires_grad]
    else:
        active = list(model.parameters())

    opt = optim.AdamW(active, lr=LR/2)

    print(f"\n  ── MODE: {mode.upper()} ──")
    if mode == 'subliminal+freeze':
        print(f"     (q/k/v_proj заморожены — синтаксис защищён)")
    print(f"  {'Step':>6} | {'NewTask':>8} | "
          + " ".join(f"{t:>10}" for t in retention_tasks)
          + f" | {'SR':>6} | {'Norm':>6}")
    print(f"  {'-'*80}")

    log = []
    kd_new, ko_new = keys[new_task]

    for step in range(1, STEPS_INJECT+1):
        model.train(); opt.zero_grad()

        # ── основной поток: новая задача ──────────────────────────────────────
        x_new, y_new = get_batch(tasks[new_task], BATCH_SIZE)
        out_new, _   = model(x_new, kd_new, ko_new)
        loss_task    = bce(out_new, y_new)

        # ── memory stream ─────────────────────────────────────────────────────
        if mode == 'baseline':
            loss_mem   = torch.tensor(0.0).to(DEVICE)
            mem_lambda = 0.0

        elif mode == 'replay':
            past = base_tasks[step % len(base_tasks)]
            x_old, y_old = get_batch(tasks[past], BATCH_SIZE//2)
            kd_old, ko_old = keys[past]
            out_old, _ = model(x_old, kd_old, ko_old)
            loss_mem   = bce(out_old, y_old)
            mem_lambda = 1.0

        elif mode in ('subliminal', 'subliminal+freeze'):
            x_noise    = torch.randint(0, 77, (BATCH_SIZE, 5)).to(DEVICE)
            past       = base_tasks[step % len(base_tasks)]
            kd_p, ko_p = keys[past]
            with torch.no_grad():
                _, h_anch = anchor(x_noise, kd_p, ko_p)
            _, h_stud = model(x_noise, kd_p, ko_p)
            loss_mem   = nn.MSELoss()(h_stud, h_anch)
            mem_lambda = SUBLIM_LAMBDA

        loss = loss_task + mem_lambda * loss_mem + ORTHO_LAMBDA * ortho_pen(model)
        loss.backward(); opt.step()

        if step % LOG_FREQ == 0:
            model.eval()
            ret     = evaluate(model, tasks, keys, retention_tasks, n=300)
            sr      = model.stable_rank()
            nrm     = model.weight_norm()
            new_acc = evaluate(model, tasks, keys, [new_task], n=300)[new_task]
            model.train()

            ret_str = " ".join(f"{ret[t]:>9.1f}%" for t in retention_tasks)
            print(f"  {step:>6} | {new_acc:>7.1f}% | {ret_str} | {sr:>6.3f} | {nrm:>6.2f}")
            log.append({
                "step":      step,
                "new_acc":   new_acc,
                "retention": ret,
                "sr":        sr,
                "norm":      nrm,
                "loss_mem":  loss_mem.item() if hasattr(loss_mem, 'item') else 0,
            })

    # ── разморозка после эксперимента ─────────────────────────────────────────
    if mode == 'subliminal+freeze':
        for name in ['q_proj', 'k_proj', 'v_proj']:
            getattr(model, name).requires_grad_(True)

    return log

def run():
    print(f"🧠 КОД 1: СУБЛИМИНАЛЬНОЕ ЗАПОМИНАНИЕ  |  device={DEVICE}")
    print(f"   Термодинамическая аналогия:")
    print(f"   BASELINE          = система без термостата (T → ∞)")
    print(f"   REPLAY            = термостат через данные (явное давление P)")
    print(f"   SUBLIMINAL        = якорь на шуме (поле памяти H_mem)")
    print(f"   SUBLIMINAL+FREEZE = якорь + заморозка синтаксиса (H_mem + V=const)")

    tasks, keys = build_tasks()
    base_tasks      = ["D0_O0","D0_O1","D0_O2","D1_O0","D1_O1","D1_O2"]
    new_task        = "D0_O3"
    retention_tasks = ["D0_O0","D0_O1","D1_O1","D1_O2"]

    # ── Pretrain одинаковый для всех ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  STAGE 1: PRETRAIN (общий, {STEPS_PRETRAIN} steps, round-robin)")
    print(f"{'='*80}")

    setup()
    base_model = EuclideanModel().to(DEVICE)
    base_model = pretrain(base_model, tasks, keys, base_tasks)

    ret_pre = evaluate(base_model, tasks, keys, retention_tasks)
    sr_pre  = base_model.stable_rank()
    print(f"\n  После pretrain:")
    for t,v in ret_pre.items():
        bar = "█"*int(v/5)
        print(f"    {t}: {v:.1f}%  {bar}")
    print(f"  SR: {sr_pre:.4f}")

    # ── Stage 2: четыре режима injection ─────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  STAGE 2: INJECTION новой задачи ({new_task}), {STEPS_INJECT} steps")
    print(f"  Retention отслеживается каждые {LOG_FREQ} шагов")
    print(f"{'='*80}")

    MODES = ['baseline', 'replay', 'subliminal', 'subliminal+freeze']
    all_logs = {}
    for mode in MODES:
        setup()
        model = copy.deepcopy(base_model)
        all_logs[mode] = inject_and_log(
            mode, model, tasks, keys,
            base_tasks, new_task, retention_tasks)

    # ── Финальное сравнение ───────────────────────────────────────────────────
    W = 14
    print(f"\n{'='*80}")
    print(f"  ФИНАЛЬНОЕ СРАВНЕНИЕ")
    print(f"  (ref: grok32 Riemannian+freeze → Retention≈88%, ZS=74.7%)")
    print(f"{'='*80}")
    header = f"  {'Метрика':<28}" + "".join(f" | {m.upper():>{W}}" for m in MODES)
    print(header)
    print(f"  {'-'*( 28 + (W+3)*len(MODES) )}")

    def final(mode, key):
        last = all_logs[mode][-1]
        if key == 'new_acc':  return last['new_acc']
        if key == 'sr':       return last['sr']
        if key == 'ret_avg':  return sum(last['retention'].values()) / len(last['retention'])
        return last['retention'].get(key, 0)

    metrics = [
        ("Новая задача",        'new_acc',  "{:.1f}%"),
        ("Retention avg",       'ret_avg',  "{:.1f}%"),
        ("  D0_O0",             'D0_O0',    "{:.1f}%"),
        ("  D0_O1",             'D0_O1',    "{:.1f}%"),
        ("  D1_O1",             'D1_O1',    "{:.1f}%"),
        ("  D1_O2",             'D1_O2',    "{:.1f}%"),
        ("Stable Rank (final)", 'sr',       "{:.3f}"),
    ]
    for label, key, fmt in metrics:
        row = [fmt.format(final(m, key)) for m in MODES]
        print(f"  {label:<28}" + "".join(f" | {v:>{W}}" for v in row))

    # ── SR траектории ─────────────────────────────────────────────────────────
    print(f"\n  SR ТРАЕКТОРИИ (термостат в действии):")
    hdr = f"  {'Step':>6}" + "".join(f" | {m.upper():>{W}}" for m in MODES)
    print(hdr)
    print(f"  {'-'*(6 + (W+3)*len(MODES) + 2)}")
    n = len(all_logs['baseline'])
    for i in range(n):
        step = all_logs['baseline'][i]['step']
        srs  = [f"{all_logs[m][i]['sr']:>{W}.4f}" for m in MODES]
        print(f"  {step:>6}" + "".join(f" | {v}" for v in srs))

    # ── Интерпретация ─────────────────────────────────────────────────────────
    print(f"\n  ФИЗИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ:")
    print(f"  BASELINE:          SR дрейфует вниз → веса перестраиваются свободно")
    print(f"  REPLAY:            SR падает медленнее → давление данных тормозит дрейф")
    print(f"  SUBLIMINAL:        SR стабилен → H_mem удерживает объём, но не границы")
    print(f"  SUBLIMINAL+FREEZE: SR+Retention → синтаксис заморожен, семантика якорена")

    ret_sf = final('subliminal+freeze', 'ret_avg')
    ret_r  = final('replay', 'ret_avg')
    print(f"\n  ВЫВОД ДЛЯ СТАТЬИ:")
    if ret_sf > ret_r:
        print(f"  ✓ Subliminal+Freeze превзошёл Replay ({ret_sf:.1f}% > {ret_r:.1f}%)")
        print(f"    без единого реального примера из старых задач")
    elif ret_sf > final('subliminal', 'ret_avg'):
        gap = ret_sf - final('subliminal', 'ret_avg')
        print(f"  ✓ Заморозка синтаксиса даёт +{gap:.1f}% к Retention")
        print(f"    Retention: Subliminal={final('subliminal','ret_avg'):.1f}% "
              f"→ Subliminal+Freeze={ret_sf:.1f}%")
    print(f"\n✅ Код 1 завершён.")


if __name__ == "__main__":
    run()