import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
import warnings

# Suppress PyTorch system warnings about complex numbers for a clean log
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
EMBED_DIM = 64
DOM_DIM = 4
OP_DIM = 6
FFN_HIDDEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
ORTHO_LAMBDA = 0.05
SUBLIMINAL_LAMBDA = 2.0  # Optimal gravity for complex MSE

class SyntaxTaskGen:
    def __init__(self, domain_idx, op_idx):
        self.domain = domain_idx; self.op = op_idx
        
    def get(self, k=50):
        a = random.randint(0, k - 1); b = random.randint(0, k - 1)
        if self.op == 0: res = (a + b) % k
        elif self.op == 1: res = abs(a - b)
        elif self.op == 2: res = max(a, b)
        elif self.op == 3: res = min(a, b)
        is_pos = random.random() > 0.5
        if not is_pos: res = (res + random.randint(1, k - 1)) % k
        op_token = 50 + self.op
        
        if self.domain == 0: p = [op_token, a, b, res, 76]
        else: p = [op_token, res, a, b, 76]
        return p, 1.0 if is_pos else 0.0

def get_batch(gen, batch_size):
    x, y = [], []
    for _ in range(batch_size):
        p, label = gen.get()
        x.append(p); y.append(label)
    return torch.LongTensor(x).to(DEVICE), torch.FloatTensor(y).unsqueeze(1).to(DEVICE)

# --- VORTEX ARCHITECTURE (COMPLEX PHASES) ---
class PhaseVortex(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(80, EMBED_DIM, dtype=torch.complex64))
        self.pos = nn.Parameter(torch.randn(5, EMBED_DIM, dtype=torch.complex64))
        self.proj_dom = nn.Linear(DOM_DIM, EMBED_DIM, bias=False)
        self.proj_op = nn.Linear(OP_DIM, EMBED_DIM, bias=False)
        
        self.q_proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False).to(torch.complex64)
        self.k_proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False).to(torch.complex64)
        self.v_proj = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False).to(torch.complex64)
        self.lin1 = nn.Linear(EMBED_DIM, FFN_HIDDEN, bias=False).to(torch.complex64)
        self.lin2 = nn.Linear(FFN_HIDDEN, EMBED_DIM, bias=False).to(torch.complex64)
        self.head = nn.Linear(EMBED_DIM, 1)

    def forward(self, x, k_dom, k_op):
        h = self.emb[x] + self.pos
        
        # Phase refraction: preserves the vector norm
        theta_dom = self.proj_dom(k_dom)
        h_dom = h * torch.complex(torch.cos(theta_dom), torch.sin(theta_dom))
        
        Q, K, V = self.q_proj(h_dom), self.k_proj(h_dom), self.v_proj(h_dom)
        attn = torch.softmax((torch.matmul(Q, K.conj().transpose(-2, -1)) / 8.0).abs(), dim=-1).to(torch.complex64)
        h_mid = h_dom + torch.matmul(attn, V)
        
        theta_op = self.proj_op(k_op)
        h_op = h_mid * torch.complex(torch.cos(theta_op), torch.sin(theta_op))
        
        ffn = torch.complex(torch.relu(self.lin1(h_op).real), torch.relu(self.lin1(h_op).imag))
        h_out = h_op + self.lin2(ffn)
        return torch.sigmoid(self.head(h_out.mean(1).abs())), h_out

def evaluate_model(model, tasks, keys_dict, title):
    model.eval()
    print(f"\n{'='*50}\n📊 {title}\n{'='*50}")
    with torch.no_grad():
        retention_acc = []
        for tname in ["D0_O0", "D0_O1", "D1_O1", "D1_O2"]:
            x_t, y_t = get_batch(tasks[tname], 1000)
            acc = ((model(x_t, *keys_dict[tname])[0] > 0.5).float() == y_t).float().mean().item() * 100
            retention_acc.append(acc)
            print(f"    💾 Retention [{tname}]: {acc:.1f}%")
        
        print(f"👉 FINAL AVERAGE RETENTION (MEMORY): {sum(retention_acc)/len(retention_acc):.1f}%")
        
        x_new, y_new = get_batch(tasks["D0_O3"], 1000)
        acc_new = ((model(x_new, *keys_dict["D0_O3"])[0] > 0.5).float() == y_new).float().mean().item() * 100
        print(f"🆕 Learning new task (Min): {acc_new:.1f}%")

def run_memory_demonstrator():
    torch.manual_seed(42)
    roots_4d = [torch.zeros(DOM_DIM).to(DEVICE) for _ in range(2)]; roots_4d[0][0] = 1.0; roots_4d[1][1] = 1.0
    deltas_6d = [torch.zeros(OP_DIM).to(DEVICE) for _ in range(4)]
    for i in range(4): deltas_6d[i][i] = 1.0

    keys_dict = {f"D{d}_O{o}": (roots_4d[d].view(1,1,-1), deltas_6d[o].view(1,1,-1)) for d in range(2) for o in range(4)}
    tasks = {name: SyntaxTaskGen(int(name[1]), int(name[4])) for name in keys_dict}

    base_tasks = ["D0_O0", "D0_O1", "D0_O2", "D1_O0", "D1_O1", "D1_O2"] # 6 starting tasks
    new_task = "D0_O3" # New task: Min
    
    print(f"🚀 STARTING DEMONSTRATOR: BATTLE FOR MEMORY | Device: {DEVICE}")
    
    # ---------------------------------------------------------
    # STAGE 1: TRAINING THE UNIFIED BASE
    # ---------------------------------------------------------
    print("\n🌀 STAGE 1: Training the base model (forming primary memory)...")
    base_model = PhaseVortex().to(DEVICE)
    opt = optim.AdamW(base_model.parameters(), lr=0.001)

    for step in range(1, 30001):
        base_model.train(); opt.zero_grad()
        name = random.choice(base_tasks)
        x, y = get_batch(tasks[name], BATCH_SIZE)
        out, _ = base_model(x, *keys_dict[name])
        loss = nn.BCELoss()(out, y) + ORTHO_LAMBDA * torch.norm(torch.matmul(base_model.proj_dom.weight.t(), base_model.proj_op.weight))
        loss.backward(); opt.step()
        if step % 10000 == 0: print(f"   Step {step}/30000 completed")

    base_state = copy.deepcopy(base_model.state_dict())
    anchor_model = copy.deepcopy(base_model).eval()

    # ---------------------------------------------------------
    # STAGE 2: THREE METHODS FOR LEARNING A NEW TASK
    # ---------------------------------------------------------

    # METHOD 1: NAIVE FINE-TUNING (Catastrophic forgetting)
    print("\n🧠 STAGE 2.1: Naive Fine-Tuning (Only new data)...")
    model_naive = PhaseVortex().to(DEVICE); model_naive.load_state_dict(base_state)
    opt_naive = optim.AdamW(model_naive.parameters(), lr=0.001)
    for _ in range(15000):
        model_naive.train(); opt_naive.zero_grad()
        x, y = get_batch(tasks[new_task], BATCH_SIZE)
        out, _ = model_naive(x, *keys_dict[new_task])
        loss = nn.BCELoss()(out, y)
        loss.backward(); opt_naive.step()

    # METHOD 2: EXPERIENCE REPLAY (Industry standard / Heavy databases)
    print("🧠 STAGE 2.2: Experience Replay (New data + Mixing with old datasets)...")
    model_replay = PhaseVortex().to(DEVICE); model_replay.load_state_dict(base_state)
    opt_replay = optim.AdamW(model_replay.parameters(), lr=0.001)
    for _ in range(15000):
        model_replay.train(); opt_replay.zero_grad()
        x_n, y_n = get_batch(tasks[new_task], BATCH_SIZE)
        out_n, _ = model_replay(x_n, *keys_dict[new_task])
        loss_n = nn.BCELoss()(out_n, y_n)
        
        past_task = random.choice(base_tasks)
        x_o, y_o = get_batch(tasks[past_task], BATCH_SIZE)
        out_o, _ = model_replay(x_o, *keys_dict[past_task])
        loss_o = nn.BCELoss()(out_o, y_o)
        
        loss = loss_n + loss_o
        loss.backward(); opt_replay.step()

    # METHOD 3: SUBLIMINAL ECHO (Vortex Method / White noise)
    print("🧠 STAGE 2.3: Subliminal Echo (New data + Pure white noise generator)...")
    model_sub = PhaseVortex().to(DEVICE); model_sub.load_state_dict(base_state)
    opt_sub = optim.AdamW(model_sub.parameters(), lr=0.001)
    for _ in range(15000):
        model_sub.train(); opt_sub.zero_grad()
        x_n, y_n = get_batch(tasks[new_task], BATCH_SIZE)
        out_n, _ = model_sub(x_n, *keys_dict[new_task])
        loss_n = nn.BCELoss()(out_n, y_n)
        
        # THE MAGIC: Replacing old datasets with absolutely random noise
        x_noise = torch.randint(0, 77, (BATCH_SIZE, 5)).to(DEVICE)
        past_task = random.choice(base_tasks)
        with torch.no_grad(): _, h_anchor = anchor_model(x_noise, *keys_dict[past_task])
        _, h_student = model_sub(x_noise, *keys_dict[past_task])
        
        loss_sub = nn.MSELoss()(h_student.real, h_anchor.real) + nn.MSELoss()(h_student.imag, h_anchor.imag)
        loss = loss_n + SUBLIMINAL_LAMBDA * loss_sub
        loss.backward(); opt_sub.step()

    # ---------------------------------------------------------
    # STAGE 3: FINAL COMPARISON (OUTPUT FOR THE ARTICLE)
    # ---------------------------------------------------------
    evaluate_model(model_naive, tasks, keys_dict, "METHOD 1: NAIVE FINE-TUNING (AMNESIA)")
    evaluate_model(model_replay, tasks, keys_dict, "METHOD 2: EXPERIENCE REPLAY (INDUSTRIAL BASELINE)")
    evaluate_model(model_sub, tasks, keys_dict, "METHOD 3: SUBLIMINAL ECHO (OUR DATA-FREE METHOD)")

if __name__ == "__main__":
    run_memory_demonstrator()
