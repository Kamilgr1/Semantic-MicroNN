# PhaseVortex: Data-Free Continual Learning & Semantic Compositionality

This repository contains proof-of-concept PyTorch experiments demonstrating new approaches to two fundamental problems in modern AI: **Catastrophic Forgetting** and **Zero-Shot Compositionality**. 

Instead of relying on massive computational power, Experience Replay, or external databases, these experiments explore how manipulating the latent space geometry (using phase keys, orthogonal penalties, and white noise) can create a robust, compositional "semantic computer" within a micro-neural network.

## Core Concepts

1. **Subliminal Echo (Data-Free Continual Learning):** Overcoming catastrophic forgetting without storing old datasets. We use pure white noise passed through frozen phase keys to maintain the topological geometry of past knowledge while learning new tasks.
2. **Crystallized Invariants (Compositional Abstraction):** Achieving zero-shot generalization by strictly decoupling "Syntax" (Attention) from "Semantics" (FFN). By freezing the syntax layers after pre-training, we force the network to abstract semantic concepts, enabling meta-composition.

---

## Experiment 1: The Memory Demonstrator (Continual Learning)
**File:** `memory_demonstrator.py`

**The Problem:** When fine-tuning a model on a new task, gradients destroy the weights of previously learned tasks (Catastrophic Forgetting). The industry standard (Experience Replay) solves this by re-training on old datasets, which is expensive and data-heavy.
**Our Solution (Subliminal Echo):** We substitute the old dataset with a `torch.randint` white noise generator. By penalizing the change in how the network "refracts" this noise through old keys, we preserve the latent geometry.

| Method | Access to Old Data | Old Tasks Retention | New Task Accuracy |
| :--- | :---: | :---: | :---: |
| **Naive Fine-Tuning** (Amnesia) | ❌ None | **49.8%** *(Random)* | 100.0% |
| **Experience Replay** (Baseline) | 💾 100% of old dataset | **96.4%** | 99.7% |
| **Subliminal Echo** (Our Method) | 🎲 **0 bytes (Pure Noise)** | **80.4%** | 98.5% |

*Result: We retain 80.4% of past knowledge using absolutely zero historical data.*

---

## Experiment 2: The Abstraction Demonstrator (Compositionality)
**File:** `abstraction_demonstrator.py`

**The Problem:** Neural networks often memorize specific token combinations instead of learning underlying abstract concepts.
**Our Solution:** We use `ORTHO_LAMBDA` to orthogonally separate domain/syntax keys from operation/logic keys. After base training, we **freeze the Attention layers**. We evaluate the network's abstraction capabilities across three cognitive levels (inspired by Vygotsky's theory of cognitive development):

| Level | Metric | Result | Interpretation |
| :--- | :--- | :--- | :--- |
| **1. Scale (Complex)** | Zero-Shot Syntax Transfer | **64.3%** avg | The network separates word order from math logic. |
| **2. Gradient (Concept)** | Zero-Shot drop when reducing train data by 3.5x | **Δ = -2.1%** | A plateau proves the network formed a strict invariant, not just interpolating data. |
| **3. Meta-Composition** | Accuracy on `SPREAD` operation via meta-key | **+30.0%** | The network synthesizes a brand new operation from two known ones. |

### The Meta-Composition Breakthrough
The network was **never trained** on the `SPREAD` operation (`max(a,b) - min(a,b)`). However, because it independently learned `MAX` and `MIN`, applying a single `k_meta='compose'` key jumps the accuracy from 51.5% (random guess) to **81.5%**. The network logically combined known invariants to solve an unseen problem.

---

## Getting Started

### Prerequisites
* Python 3.8+
* PyTorch (CUDA recommended but runs on CPU)

### Running the Demonstrators
Clone the repository and run the scripts directly. They are fully autonomous and generate the benchmark tables in the console.

```bash
git clone [https://github.com/yourusername/PhaseVortex.git](https://github.com/yourusername/PhaseVortex.git)
cd PhaseVortex

# Run the Continual Learning benchmark
python memory_demonstrator.py

# Run the Zero-Shot Compositionality benchmark
python abstraction_demonstrator.py
