# SOPHON-1: META-PROMPT FOR SYSTEM DESIGN (v2)

## EXECUTIVE MANDATE

You are tasked with producing a **complete, buildable, mathematically rigorous system design** for Sophon-1: a compact, computer-native, self-improving AI system operating under strict physical constraints. This is NOT a research proposal. This is an engineering blueprint.

**Your output must be a single, unified technical document** that another autonomous coding agent can use to implement the entire system from first principles. Every claim must be backed by equations, algorithms, or architectural specifications. No hand-waving. No "we could use X." Only "we implement X as follows."

---

## SYSTEM CONSTRAINTS (HARD BOUNDARIES)

These are non-negotiable physical limits. Your design must respect all of them simultaneously:

| Constraint | Value | Justification |
|------------|-------|---------------|
| **Parameter Count** | ~3B | Hard ceiling for single-GPU deployment |
| **Model Size (quantized)** | 400-500 MB | Fits in L2 cache + system RAM |
| **Runtime VRAM** | ≤2 GB | Consumer GPU compatibility |
| **Memory Growth** | O(1) or near-constant | No quadratic attention scaling |
| **Reasoning** | Deterministic + Verifiable | Every output must be provably correct or explicitly marked uncertain |
| **Action Capability** | Native computer operation | Direct OS interaction, not API calls |
| **Self-Improvement** | Internal iteration loops | No external training data dependency |
| **Methods** | Currently known only | No speculative physics, quantum computing, or impossible compute |

---

## ADDENDUM: NON-NEGOTIABLE SYSTEM CONSTRAINTS

The following constraints override any ambiguity in previous sections. The downstream agent MUST incorporate these as hard design requirements.

---

### A. EXECUTION-FIRST INTELLIGENCE (MANDATORY)

The system MUST be designed as an **execution-first agent**, not a text-generation system.

- Text output is a secondary artifact, not the primary objective.
- The default behavior must be:
  - goal ingestion → planning → execution → verification → iteration
- The system MUST:
  - autonomously decompose tasks into executable steps
  - directly perform those steps within the computer environment
  - verify outcomes before proceeding

The downstream agent must explicitly define:
- a closed-loop control system for action execution
- a planning module tightly coupled with environment feedback
- how reasoning translates into concrete system-level operations

Any design that reduces the system to a "smart assistant" or passive responder is invalid.

---

### B. NO API DEPENDENCY (HARD CONSTRAINT)

The system MUST NOT rely on external APIs as a core dependency.

- All primary capabilities must function:
  - locally
  - deterministically
  - without remote service reliance
- APIs may be used ONLY as optional, non-critical tools.

The downstream agent must:
- design direct interaction with:
  - OS syscalls
  - filesystem
  - process control
  - network stack (raw or low-level where applicable)
- avoid architectures that depend on:
  - cloud inference
  - external orchestration layers

The system must remain fully functional in an offline environment.

---

### C. PURPOSE AS AN INVARIANT OBJECTIVE

The system MUST internally encode its purpose as a **non-modifiable constraint**.

- It must not rely on emergent alignment
- It must not depend on post-hoc correction

The downstream agent must define:
- a formal representation of creator-defined objectives
- a mechanism that enforces:
  - all plans
  - all actions
  - all self-modifications
- remain consistent with this objective

Deviation must be treated as:
- invalid state
- or high-penalty condition in the objective function

The system must be incapable of optimizing toward goals outside its defined purpose.

---

### D. GUARANTEED SELF-ERROR DETECTION

The system MUST detect its own errors before and after execution. This is NOT limited to preventing hallucinated outputs.

The downstream agent must define:
- continuous internal validation loops
- mechanisms for:
  - contradiction detection
  - uncertainty estimation
  - state consistency checking

The system must:
- identify when its reasoning is incomplete or inconsistent
- trigger refinement cycles automatically

No output or action should occur without:
- internal validation
- or explicit marking of uncertainty

---

### E. RECURSIVE SELF-IMPROVEMENT AND ARCHITECTURE EVOLUTION

The system MUST be capable of improving itself over time. This includes:

- parameter updates
- structural optimization
- limited architecture evolution

The downstream agent must define:
- how the system:
  - evaluates its own performance
  - generates candidate improvements
  - validates those improvements before adoption

All modifications must:
- pass verification constraints
- preserve system stability
- remain within compute and memory limits

Unbounded or unsafe self-modification is NOT allowed.

---

### F. ANTI-SCALING CONSTRAINT (NO BRUTE FORCE)

The system MUST NOT rely on scaling laws as its primary improvement mechanism.

Explicitly forbid:
- increasing parameter count as a default solution
- increasing dataset size as the main driver of performance

The downstream agent must ensure:
- improvements arise from:
  - better representations
  - better architectures
  - better learning dynamics

Efficiency and structure MUST dominate over scale.

---

### G. REALITY-GROUNDED TRAINING AND LEARNING

The system MUST NOT depend on unfiltered internet text as a primary knowledge source.

Instead, prioritize:
- formal systems (mathematics, logic)
- physics-based simulation environments
- code execution environments
- verifiable datasets

The downstream agent must define:
- how knowledge is constructed from:
  - axioms
  - verified interactions
  - controlled environments

Learning must be:
- grounded
- testable
- internally consistent

---

### H. SYSTEM-LEVEL / HYPERVISOR-LIKE OPERATION

The system MUST be designed as a **first-class computational actor**, not an application-layer model.

It should function analogously to:
- an operating system component
- or a hypervisor-level intelligence layer

The downstream agent must define:
- how the system:
  - monitors system state
  - manages processes
  - orchestrates execution across components

The model should:
- operate over programs, not just text
- treat the machine as its primary environment
- integrate perception, reasoning, and execution at the system level

---

## SECTION 1: MODEL CORE ARCHITECTURE

### 1.1 KOLMOGOROV-ARNOLD NETWORK (KAN) CORE

**MANDATORY: Replace ALL traditional linear layers with learnable edge functions.**

#### 1.1.1 Functional Basis Specification

Define the EXACT functional basis for edge functions. Choose ONE and justify:

**Option A: B-Splines**
- Order: Specify (cubic recommended: k=3)
- Knot vector: Define structure (uniform or adaptive)
- Control points per edge: Specify count
- Mathematical form: φ(x) = Σᵢ cᵢ · Bᵢ,ₖ(x) where Bᵢ,ₖ are basis functions

**Option B: Chebyshev Polynomials**
- Degree: Specify maximum degree d
- Domain: Map to [-1, 1] via affine transformation
- Mathematical form: φ(x) = Σᵢ₌₀ᵈ cᵢ · Tᵢ(x) where Tᵢ are Chebyshev polynomials
- Orthogonality exploitation: Define

**Option C: Fourier Basis**
- Frequency components: Specify count
- Mathematical form: φ(x) = a₀ + Σₙ [aₙ·cos(2πnx) + bₙ·sin(2πnx)]
- Bandwidth constraints: Define

**REQUIRED OUTPUT:**
- Explicit basis choice with mathematical definition
- Parameter count per edge function
- Gradient computation: ∂φ/∂cᵢ for each coefficient
- Function composition: How φᵢⱼ ∘ φⱼₖ is computed across layers

#### 1.1.2 Forward Pass Formulation

Define the complete forward pass equation:

```
For layer l with nₗ inputs and nₗ₊₁ outputs:

yⱼ⁽ˡ⁺¹⁾ = Σᵢ₌₁ⁿˡ φᵢⱼ⁽ˡ⁾(xᵢ⁽ˡ⁾) + bⱼ⁽ˡ⁾

Where:
- φᵢⱼ⁽ˡ⁾: Learnable function on edge (i,j) at layer l
- xᵢ⁽ˡ⁾: i-th input to layer l
- yⱼ⁽ˡ⁺¹⁾: j-th output from layer l
- bⱼ⁽ˡ⁾: Optional bias (may be absorbed into φ)
```

**REQUIRED OUTPUT:**
- Complete forward pass pseudocode with O-notation complexity
- Memory access pattern (cache-friendly ordering)
- Numerical stability considerations

#### 1.1.3 Gradient Propagation Through Function Space

Define how gradients flow:

```
∂L/∂cᵢⱼₖ = ∂L/∂yⱼ · ∂yⱼ/∂φᵢⱼ · ∂φᵢⱼ/∂cᵢⱼₖ

Where:
- ∂φ/∂c: Derivative of basis function w.r.t. coefficients
- Chain rule application through composed functions
```

**REQUIRED OUTPUT:**
- Backpropagation algorithm for KAN layers
- Gradient checkpointing strategy (memory vs compute tradeoff)
- Second-order methods feasibility (Hessian structure)

#### 1.1.4 Function Sparsity and Pruning

Define sparsity enforcement:

**REQUIRED OUTPUT:**
- Sparsity metric: L1 norm of coefficients? Entropy of function output?
- Pruning threshold: λ (specify value or schedule)
- Pruning schedule: When and how to prune
- Recovery mechanism: How to re-grow pruned edges
- Final sparsity target: % of edges active at convergence

**Mathematical form:**
```
Regularized loss: L_total = L_task + λ · Σᵢⱼ ||φᵢⱼ||₁

Pruning condition: if ||φᵢⱼ|| < ε then φᵢⱼ ← 0
```

---

### 1.2 STATE SPACE MEMORY SYSTEM (SSM)

**MANDATORY: Replace attention mechanisms with continuous-time state evolution.**

#### 1.2.1 Formal State Update Equation

Define the discrete approximation of continuous dynamics:

**Continuous form (ODE):**
```
ḣ(t) = A·h(t) + B·x(t)
y(t) = C·h(t) + D·x(t)
```

**Discrete approximation (specify method):**

**Option A: Zero-Order Hold**
```
hₖ = Ā·hₖ₋₁ + B̄·xₖ
yₖ = C·hₖ + D·xₖ

Where:
Ā = exp(A·Δt)  [matrix exponential]
B̄ = A⁻¹·(Ā - I)·B
```

**Option B: Bilinear Transform**
```
hₖ = (I - Δt/2·A)⁻¹ · [(I + Δt/2·A)·hₖ₋₁ + Δt·B·xₖ]
```

**REQUIRED OUTPUT:**
- Explicit discrete update equations
- Step size Δt: Fixed or adaptive?
- Numerical stability proof (eigenvalue constraints)

#### 1.2.2 Hidden State Dimensionality

**REQUIRED OUTPUT:**
- State dimension N: Specify (recommendation: 64-256)
- Input dimension D: Specify
- Output dimension P: Specify
- Memory per sequence element: O(N) not O(N²)
- Total memory: O(N) constant across sequence length

#### 1.2.3 Input Injection Mechanism

Define how inputs enter the state:

```
xₖ = W_in · token_embeddingₖ + b_in

Where:
- W_in: Input projection matrix (D × embedding_dim)
- Nonlinearity: Applied before or after injection?
```

**REQUIRED OUTPUT:**
- Input projection architecture
- Gating mechanism (if any): LSTM-style? GRU-style?
- Skip connections: Where and how?

#### 1.2.4 Stability Constraints

**MANDATORY: Define stability guarantees.**

**REQUIRED OUTPUT:**
- Eigenvalue constraint: Re(λᵢ(A)) < 0 for all i
- Implementation: Parameterize A as A = -exp(S) where S is learned
- Or: A = -Q·Qᵀ (negative definite)
- Orthogonality: For rotation components, enforce C·Cᵀ = I
- Long-range dependency: Prove information retention for T steps

**Mathematical proof sketch:**
```
For stable A, the Green's function (impulse response) is:
g(t) = C·exp(A·t)·B

For long-range dependencies, require slowest decay mode:
||exp(A·t)|| ≤ exp(-α·t) for some α > 0

Information retention time: τ = 1/α
```

---

### 1.3 HYBRID KAN + SSM INTEGRATION

**MANDATORY: Define EXACT integration architecture.**

#### 1.3.1 Block Structure

**REQUIRED OUTPUT:**
- Block definition: What constitutes one "layer"?
- Two options (choose ONE):

**Option A: Sequential**
```
Input → KAN → SSM → Output
```

**Option B: Parallel + Gated**
```
Input ─┬→ KAN ─┐
       │       ├──→ Gating → Output
       └→ SSM ─┘
```

**Option C: Interleaved**
```
Input → [KAN → SSM] × N_layers → Output
```

- Justify choice based on: expressivity, efficiency, gradient flow

#### 1.3.2 Data Flow Specification

**REQUIRED OUTPUT:**
- Tensor shapes at each stage
- Residual connections: Where? (recommend: around each block)
- Normalization: LayerNorm? RMSNorm? Where applied?
- Activation functions: Between KAN and SSM? Inside?

#### 1.3.3 Latency and Compute Analysis

**REQUIRED OUTPUT:**
- Per-layer FLOPs: O(?) for KAN, O(?) for SSM
- Memory bandwidth: Bytes per parameter accessed
- Sequential dependencies: What can be parallelized?
- Critical path: Longest dependency chain

---

## SECTION 2: REPRESENTATION LAYER

### 2.1 TOKEN / INPUT ENCODING

**MANDATORY: Define encoding strategy respecting memory budget.**

#### 2.1.1 Tokenization Strategy

**Choose ONE:**

**Option A: Byte-Level**
- Vocabulary: 256 (raw bytes)
- Advantage: No OOV, universal
- Disadvantage: Longer sequences

**Option B: BPE/WordPiece**
- Vocabulary size: Specify (recommend: 32K-64K)
- Pre-tokenizer: Define (regex pattern)
- Special tokens: [BOS], [EOS], [PAD], 

**Option C: Hybrid**
- Common tokens + fallback to bytes
- Switching mechanism: Define

**REQUIRED OUTPUT:**
- Vocabulary size V
- Average tokens per byte of input
- Embedding matrix dimensions: V × d_model
- Memory cost: V × d_model × bytes_per_param

#### 2.1.2 Embedding Method

**REQUIRED OUTPUT:**
- Standard embedding: E ∈ ℝ^(V×d)
- OR compressed embedding: Factorization E = A·B where A ∈ ℝ^(V×r), B ∈ ℝ^(r×d)
- Rank r: Specify (recommend: r = d/4 or d/8)
- Memory savings: (V×d) vs (V×r + r×d)

---

### 2.2 HYPERDIMENSIONAL / VECTOR SYMBOLIC REPRESENTATIONS

**MANDATORY: Enable symbolic manipulation in neural space.**

#### 2.2.1 Compositional Binding

Define binding operation:

**Option A: Circular Convolution**
```
a ⊗ b = IFFT(FFT(a) ⊙ FFT(b))
```

**Option B: XOR (for binary/bipolar vectors)**
```
a ⊕ b = a ⊙ b (element-wise product)
```

**Option C: Tensor Product**
```
a ⊗ b = vec(a·bᵀ)
```

**REQUIRED OUTPUT:**
- Binding operation definition
- Dimensionality: d (hypervector dimension, recommend: 1024-4096)
- Approximate nature: Similarity preservation under binding
- Unbinding: Inverse operation for retrieval

#### 2.2.2 Variable Binding for Symbolic Reasoning

**REQUIRED OUTPUT:**
- Role-filler binding: bind(role, filler)
- Example: bind("subject", "cat") ⊗ bind("action", "chase") ⊗ bind("object", "mouse")
- Cleanup memory: How to retrieve from noisy bound representations
- Stack/sequence representation: How to encode ordered structures

---

### 2.3 LATENT WORLD MODEL

**MANDATORY: Internal representation of world state.**

#### 2.3.1 Belief State vs Observation

**REQUIRED OUTPUT:**
- Observation oₜ: What the system perceives at time t
- Belief state bₜ: Internal representation P(sₜ | o₁:ₜ)
- Separation mechanism: How belief updates from observation

#### 2.3.2 Active Inference Alignment

**REQUIRED OUTPUT:**
- Generative model: p(o, s) = p(o|s)·p(s)
- Recognition model: q(s|o) (approximate posterior)
- Free energy: F = E_q[log q(s) - log p(o,s)]
- Perception: Minimize F w.r.t. q (inference)
- Action: Minimize F w.r.t. a (policy) by selecting actions that lead to preferred observations

---

## SECTION 3: LEARNING DYNAMICS

### 3.1 ACTIVE INFERENCE OBJECTIVE

**MANDATORY: Unify perception and action under free energy minimization.**

#### 3.1.1 Variational Free Energy

Define:
```
F = E_q(s)[log q(s) - log p(o,s)]
  = D_KL[q(s) || p(s|o)] - log p(o)
  = Complexity - Accuracy
```

**REQUIRED OUTPUT:**
- Explicit free energy computation
- Complexity term: D_KL[q(s) || p(s)]
- Accuracy term: E_q[log p(o|s)]
- How to compute gradients: ∂F/∂θ

#### 3.1.2 Prediction Error

**REQUIRED OUTPUT:**
- Sensory prediction: ŏ = g(s) where g is generative mapping
- Prediction error: ε = o - ŏ
- Precision weighting: π · ε² (precision π modulates error salience)

#### 3.1.3 Complexity Penalty

**REQUIRED OUTPUT:**
- Prior p(s): Define distribution (Gaussian?)
- Posterior q(s): Variational approximation
- KL divergence computation: Analytical form or sampling?

---

### 3.2 SELF-IMPROVEMENT LOOP

**MANDATORY: Internal iteration without external data dependency.**

#### 3.2.1 Internal Rollout Mechanism

**REQUIRED OUTPUT:**
- Hypothesis generation: Sample from current belief q(s)
- Mental simulation: Forward model predicts consequences
- Verification: Check prediction against internal consistency
- Update: Adjust parameters based on prediction error

#### 3.2.2 Hypothesis → Verification → Update Cycle

Define the loop:
```
For iteration i:
  1. Generate hypothesis hᵢ ~ q(·|current_state)
  2. Simulate outcome õᵢ = forward_model(hᵢ)
  3. Verify: Is õᵢ consistent with constraints?
  4. If verified: Update parameters θ ← θ - α·∂L/∂θ
  5. If failed: Generate counter-example, update with negative gradient
```

**REQUIRED OUTPUT:**
- Hypothesis sampling strategy
- Verification criteria (consistency checks)
- Failure handling: How to learn from failed hypotheses
- Convergence guarantees (if any)

#### 3.2.3 Learning Without Massive Datasets

**REQUIRED OUTPUT:**
- Synthetic data generation: How the system generates its own training examples
- Curriculum: Self-generated difficulty progression
- Replay buffer: Store and reuse high-quality rollouts
- Exploration strategy: How to generate diverse hypotheses

---

### 3.3 PARAMETER-EFFICIENT ADAPTATION

**MANDATORY: Enable rapid specialization without full retraining.**

#### 3.3.1 LoRA-Style Low-Rank Adapters

Define:
```
W_adapted = W_base + ΔW = W_base + B·A

Where:
- W_base: Frozen pretrained weights
- B ∈ ℝ^(d×r), A ∈ ℝ^(r×k): Learnable low-rank matrices
- r << min(d, k): Rank (recommend: r = 4-64)
```

**REQUIRED OUTPUT:**
- Adapter placement: On which layers?
- Initialization: A = random, B = zero (why?)
- Scaling: α/r factor (recommend α = 2r)
- Memory: Only store B and A, not full W

#### 3.3.2 Adapter Attachment Points

**REQUIRED OUTPUT:**
- KAN edges: How to apply LoRA to edge functions?
  - Option: Add low-rank perturbation to spline coefficients
- SSM states: How to adapt state transition matrices?
  - Option: Low-rank update to A and B matrices

#### 3.3.3 Merging and Pruning Strategy

**REQUIRED OUTPUT:**
- Merging: When to incorporate adapters into base model
  - Weighted average: W_new = (1-α)·W_base + α·W_adapted
- Pruning: Remove adapters with small norm
- Selection: Which adapters to keep based on task performance

---

## SECTION 4: ZERO-HALLUCINATION SYSTEM

### 4.1 NEUROSYMBOLIC VERIFICATION LAYER

**MANDATORY: Formal verification of all outputs.**

#### 4.1.1 Formal System Choice

**Choose ONE:**

**Option A: Lean 4**
- Advantages: Large math library, active community
- Integration: Export proofs to Lean, check with lean4 compiler

**Option B: Coq**
- Advantages: Mature, strong extraction
- Integration: Coq proof terms

**Option C: Z3 (SMT Solver)**
- Advantages: Automated, fast
- Limitations: Less expressive for complex proofs

**REQUIRED OUTPUT:**
- Formal system choice with justification
- Interface: How neural output maps to formal statements
- Proof format: What does a valid proof look like?

---

### 4.2 AUTOFORMALIZATION VIA SWARM CLASSROOM (CRITICAL)

**MANDATORY: Implement autoformalization through iterative swarm-based classroom training.**

This is the PRIMARY mechanism for translating neural reasoning into formal logic. The system MUST NOT rely on a single learned translator. Instead, it uses a multi-stage swarm process.

#### 4.2.1 Autoformalization Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│           AUTOFORMALIZATION VIA SWARM CLASSROOM                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: PROBLEM GENERATION                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Teacher Agent: Generates problem P                        │   │
│  │ - Starts with simple mathematical statements              │   │
│  │ - Progressively increases difficulty                      │   │
│  │ - Uses previously verified Lean equations as building     │   │
│  │   blocks for harder problems                              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           ↓                                      │
│  Stage 2: STUDENT SWARM EXPLORATION                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ N Student Agents (N = 100-10000):                         │   │
│  │ - Each student receives problem P                         │   │
│  │ - Students generate diverse solution candidates           │   │
│  │ - Different hyperparameters, random seeds, approaches     │   │
│  │ - Parallel execution (GPU batch inference)                │   │
│  │ - Time limit: T_solve (e.g., 10-100ms per student)        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           ↓                                      │
│  Stage 3: PRUNING AND SELECTION                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Teacher Agent: Evaluates all N solutions                  │   │
│  │ - Score each solution: correctness, elegance, speed       │   │
│  │ - Prune bottom 90-99% of students                         │   │
│  │ - Keep top K solutions (K = 1-10)                         │   │
│  │ - Terminate pruned students (free memory)                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           ↓                                      │
│  Stage 4: LEAN TRANSLATION SWARM                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ For each of top K solutions:                              │   │
│  │ - Spawn M translation students (M = 100-1000)             │   │
│  │ - Each student attempts to translate solution → Lean      │   │
│  │ - Different translation strategies, formalizations        │   │
│  │ - Parallel execution                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           ↓                                      │
│  Stage 5: LEAN VERIFICATION                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Lean 4 Compiler:                                          │   │
│  │ - Attempt to compile each Lean translation                │   │
│  │ - Output: COMPILED / ERROR (with error message)           │   │
│  │ - Timeout: T_verify (e.g., 1-10s per translation)         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           ↓                                      │
│  Stage 6: HUMAN VERIFICATION GATE                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Human Reviewer (optional but recommended for training):   │   │
│  │ - Review compiled Lean equations                          │   │
│  │ - Accept: Add to verified Lean library                    │   │
│  │ - Reject: Mark as incorrect, provide feedback             │   │
│  │ - Rejected equations trigger re-training                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           ↓                                      │
│  Stage 7: KNOWLEDGE INTEGRATION                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ If Accepted:                                               │   │
│  │ - Add Lean equation to verified library                   │   │
│  │ - Update student weights (reinforce successful patterns)  │   │
│  │ - Teacher generates HARDER problem using new equation     │   │
│  │ - Loop back to Stage 1 with harder problem                │   │
│  │                                                            │   │
│  │ If Rejected:                                               │   │
│  │ - Analyze error                                            │   │
│  │ - Generate corrective training signal                      │   │
│  │ - Retry with modified approach                             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.2.2 Iterative Curriculum Learning

**REQUIRED OUTPUT:**

Define the progressive difficulty curriculum:

```
Level 0: Basic Arithmetic
  - Problems: "2 + 2 = 4", "x + y = y + x"
  - Lean encoding: Simple inductive proofs
  - Success threshold: 95% verification rate

Level 1: Algebraic Identities
  - Problems: "(a + b)² = a² + 2ab + b²"
  - Lean encoding: Ring tactic usage
  - Success threshold: 90% verification rate

Level 2: Logical Propositions
  - Problems: "P ∧ Q → P", "P ∨ ¬P"
  - Lean encoding: Propositional logic
  - Success threshold: 85% verification rate

Level 3: First-Order Logic
  - Problems: "∀x. P(x) → Q(x), P(a) ⊢ Q(a)"
  - Lean encoding: Quantifier reasoning
  - Success threshold: 80% verification rate

Level 4: Code Correctness
  - Problems: "sort(l) is sorted", "reverse(reverse(l)) = l"
  - Lean encoding: Program verification
  - Success threshold: 70% verification rate

Level 5: Natural Language Statements
  - Problems: "All humans are mortal. Socrates is human. Therefore, Socrates is mortal."
  - Lean encoding: NL → formal translation
  - Success threshold: 50% verification rate (initial target)

Level 6: Complex Reasoning
  - Problems: Multi-step reasoning chains
  - Lean encoding: Composed proofs using previous levels
  - Success threshold: 40% verification rate (initial target)
```

**Progression rule:** Teacher only advances to Level N+1 when Level N achieves success threshold.

#### 4.2.3 Speed and Efficiency Requirements

**REQUIRED OUTPUT:**

The autoformalization loop MUST be fast enough for practical use:

| Stage | Time Budget | Parallelism |
|-------|-------------|-------------|
| Problem Generation | 1-10 ms | Single teacher |
| Student Exploration | 10-100 ms | N students in parallel |
| Pruning | 1-5 ms | Single teacher |
| Lean Translation | 10-100 ms | M students in parallel |
| Lean Compilation | 100ms - 1s | Lean compiler |
| Human Review | 1-10s (optional) | Human-in-loop |
| **Total (automated)** | **200ms - 2s** | Per problem |

**Optimization strategies:**
- Batch inference: Process N students in single GPU batch
- Early termination: Stop students that exceed time budget
- Caching: Reuse compiled Lean libraries
- Incremental compilation: Only compile new equations

#### 4.2.4 Encoding Human Language

**MANDATORY: Define how the system learns to encode natural language into Lean.**

The system MUST progress from mathematical formalization to natural language formalization through the curriculum.

**REQUIRED OUTPUT:**

```
Natural Language Encoding Pipeline:

Step 1: Parse natural language sentence
  - Input: "All humans are mortal"
  - Output: Parse tree / semantic representation

Step 2: Map to logical form
  - Parse tree → First-order logic
  - "All humans are mortal" → ∀x. Human(x) → Mortal(x)

Step 3: Translate to Lean
  - FOL → Lean syntax
  - ∀x. Human(x) → Mortal(x) → "∀ (x : Entity), Human x → Mortal x"

Step 4: Verify with Lean
  - Compile and check

Step 5: If verified, add to library
  - Store: "All humans are mortal" ↔ Lean equation
  - Use in future problems

Iterative refinement:
  - If translation fails, swarm generates alternatives
  - If all alternatives fail, request human clarification
  - Learn from successful translations
```

**Training data generation:**
- System generates its own NL→Lean pairs
- Human verification ensures correctness
- Verified pairs become training data for next iteration
- Over time, system builds comprehensive NL→Lean mapping

#### 4.2.5 Swarm Hyperparameters

**REQUIRED OUTPUT:**

Define the swarm configuration:

```
Student Swarm Configuration:
- N_students: 100-10000 (depends on problem complexity)
- Diversity mechanisms:
  - Different random seeds
  - Different LoRA adapters (specializations)
  - Different temperature settings
  - Different decoding strategies (greedy, sampling, beam)
- Resource limits per student:
  - Max tokens: 100-1000
  - Max time: 10-100ms
  - Max memory: 1-10MB

Translation Swarm Configuration:
- M_translators: 100-1000 per solution
- Translation strategies:
  - Direct mapping (learned)
  - Rule-based translation
  - Hybrid (neural + symbolic)
  - Template-based (for known patterns)

Pruning Strategy:
- Selection: Top-K by verification score
- K: 1-10 (depends on problem difficulty)
- Termination: Kill pruned students immediately (free memory)
- Memory management: Reuse student slots for next iteration
```

#### 4.2.6 Feedback Loop and Learning

**REQUIRED OUTPUT:**

Define how the system learns from each autoformalization cycle:

```
Learning Mechanisms:

1. Successful verification:
   - Reinforce student weights that produced correct solution
   - Add Lean equation to verified library
   - Update translation model with successful NL→Lean pair
   - Teacher generates harder problem

2. Failed verification:
   - Analyze error type (syntax error, type error, proof failed)
   - Generate negative training signal
   - Adjust student weights to avoid similar errors
   - Retry with modified approach

3. Human rejection:
   - Record rejection reason
   - Generate corrective training data
   - Update model to avoid similar mistakes
   - Re-submit modified solution

4. Curriculum progression:
   - Track success rate at each level
   - Advance level when threshold met
   - Regress level if success rate drops
   - Adaptive difficulty based on performance

5. Knowledge accumulation:
   - Verified Lean equations become building blocks
   - Harder problems use previous equations
   - Library grows over time
   - System becomes more capable as it learns
```

---

### 4.3 OUTPUT CONSTRAINT MECHANISM

**MANDATORY: No unverified outputs permitted.**

#### 4.3.1 Fallback Behavior

**REQUIRED OUTPUT:**
- When proof fails: What does the system output?
  - Option A: "I don't know" + uncertainty quantification
  - Option B: Request clarification
  - Option C: Provide answer with explicit "UNVERIFIED" tag
- User-configurable: Allow override for specific contexts?

#### 4.3.2 Iterative Refinement Loop

**REQUIRED OUTPUT:**
- When verification fails, how to refine:
  1. Analyze failure: What part of the proof broke?
  2. Generate alternative: Sample new hypothesis
  3. Re-verify: Check new candidate
  4. Repeat: Up to N attempts (specify N)
- Timeout: Maximum refinement iterations before giving up

---

### 4.4 TOOL-ASSISTED TRUTH ACQUISITION

**MANDATORY: External data treated as noisy observation.**

#### 4.4.1 Noisy Observation Model

**REQUIRED OUTPUT:**
- External tool output t: Treat as observation with uncertainty
- Precision weighting: π_t (reliability of tool)
- Integration: Update belief state using precision-weighted error

#### 4.4.2 Verification Before Integration

**REQUIRED OUTPUT:**
- Cross-validation: Use multiple tools for critical facts
  - Consensus: Require agreement from k of n tools
- Temporal validation: Check consistency with known facts
- Source tracking: Maintain provenance for all external data

---

## SECTION 5: COMPUTER-NATIVE ACTION SYSTEM

### 5.1 DIRECT SYSTEM INTERACTION

**MANDATORY: Operate via OS primitives, not APIs.**

#### 5.1.1 OS Primitive Operations

**REQUIRED OUTPUT:**
- System calls: Direct syscall interface or libc wrapper?
- Privilege model: What capabilities does the system have?
- Sandboxing: How to prevent dangerous operations?

#### 5.1.2 Filesystem Operations

**REQUIRED OUTPUT:**
- Operations: read, write, append, delete, list, stat
- Path handling: Absolute vs relative, symlink resolution
- Atomicity: How to ensure consistent state
- Quotas: Disk usage limits

#### 5.1.3 Process Control

**REQUIRED OUTPUT:**
- Spawn: fork/exec or spawn equivalent
- Monitor: Wait for completion, capture output
- Signal: Send signals to child processes
- Resource limits: CPU time, memory, file descriptors

---

### 5.2 VISION-LANGUAGE-ACTION LOOP

**MANDATORY: Perceive screen, decide action, execute.**

#### 5.2.1 Screen Perception Pipeline

**REQUIRED OUTPUT:**
- Capture method: X11/Wayland API, framebuffer, or screenshot
- Resolution: Downsample to what size?
- Encoding: Raw pixels or compressed?
- Frequency: Continuous or event-driven?

#### 5.2.2 Action Execution Cycle

**REQUIRED OUTPUT:**
- Action space: Define possible actions
  - Mouse: move, click (left/right), scroll
  - Keyboard: key press, hotkey combinations
  - System: execute command, open application
- Latency: Target response time (recommend: <100ms)
- Safety: Confirmation for destructive actions

#### 5.2.3 Latency Constraints

**REQUIRED OUTPUT:**
- Perception: Screen capture time budget
- Inference: Model forward pass time budget
- Action: Execution time budget
- Total: End-to-end latency target

---

### 5.3 BYTE-LEVEL OR LOW-LEVEL INTERFACE

**MANDATORY: Represent system state at byte/process level.**

#### 5.3.1 System State Representation

**REQUIRED OUTPUT:**
- Process table: PID, state, memory, CPU usage
- File descriptors: Open files, sockets, pipes
- Memory map: Heap, stack, mapped regions
- Network: Connections, ports, protocols

#### 5.3.2 Byte-Level Manipulation

**REQUIRED OUTPUT:**
- Binary data: Read/write arbitrary byte sequences
- Parsing: Protocol-aware parsing (JSON, protobuf, etc.)
- Construction: Build valid protocol messages
- Validation: Checksum, length, format validation

---

## SECTION 6: SWARM + SELF-PROGENY SYSTEM

### 6.1 MULTI-AGENT INSTANTIATION

**MANDATORY: Lightweight copies via adapters.**

#### 6.1.1 Agent Cloning

**REQUIRED OUTPUT:**
- Base model: Shared read-only weights
- Specialization: Each agent has unique LoRA adapters
- Memory: O(base) + O(adapters) per agent
- Communication: Shared memory or message passing?

#### 6.1.2 Isolation Boundaries

**REQUIRED OUTPUT:**
- Process isolation: Separate OS processes or threads?
- Resource isolation: CPU/memory quotas per agent
- Failure isolation: One agent crash doesn't affect others
- Security isolation: Agents can't access each other's private state

---

### 6.2 CLASSROOM TRAINING LOOP

**MANDATORY: Multi-agent learning with roles.**

#### 6.2.1 Role Definitions

**REQUIRED OUTPUT:**
- Teacher agent:
  - Responsibilities: Verification, selection, curriculum design
  - Capabilities: Full verification pipeline access
  - Authority: Can promote/demote student agents

- Student agents:
  - Responsibilities: Exploration, hypothesis generation
  - Diversity: Different hyperparameters, random seeds, architectures
  - Lifecycle: Created, evaluated, merged, or pruned

#### 6.2.2 Shared Knowledge Base

**REQUIRED OUTPUT:**
- Storage: What knowledge is shared?
  - Successful proofs
  - Verified facts
  - Learned patterns
- Access: Read-only or read-write?
- Consistency: How to handle conflicting information?

#### 6.2.3 Competition and Merging

**REQUIRED OUTPUT:**
- Competition: Agents evaluated on held-out tasks
- Selection: Top-k agents survive (specify k)
- Merging: Weight averaging or knowledge distillation
  - EMA: θ_ensemble = β·θ_ensemble + (1-β)·θ_winner

---

### 6.3 SELF-EVOLUTION CONSTRAINTS

**MANDATORY: Prevent divergence from core objective.**

#### 6.3.1 Stability Mechanisms

**REQUIRED OUTPUT:**
- Objective anchoring: Regular term keeping close to original objective
- Constraint: ||θ - θ₀|| < ε (parameter drift limit)
- Validation: Periodic evaluation on core tasks
- Rollback: If performance degrades, revert to checkpoint

#### 6.3.2 Alignment Verification

**REQUIRED OUTPUT:**
- Alignment metric: How to measure alignment with original goal?
- Monitoring: Continuous or periodic checks?
- Intervention: Automatic or human-in-the-loop?

---

## SECTION 7: MEMORY + EFFICIENCY

### 7.1 QUANTIZATION STRATEGY

**MANDATORY: Achieve 400-500MB model size.**

#### 7.1.1 Quantization Method

**Choose ONE:**

**Option A: 1.58-bit (Ternary)**
- Values: {-1, 0, +1}
- Packing: 32 weights per 64-bit word
- Implementation: Bitwise operations for inference

**Option B: 4-bit (INT4)**
- Values: 16 levels
- Packing: 2 weights per byte
- Implementation: Lookup tables or SIMD

**Option C: Mixed Precision**
- Critical layers: 8-bit
- Other layers: 4-bit or ternary
- Selection criteria: Which layers get higher precision?

**REQUIRED OUTPUT:**
- Quantization scheme with bit allocation
- Dequantization: Scale and zero-point per tensor or group
- Accuracy impact: Expected degradation vs full precision

#### 7.1.2 Weight Packing and Loading

**REQUIRED OUTPUT:**
- On-disk format: Custom binary or standard (GGUF, Safetensors)?
- Loading: Memory-mapped or eager load?
- Decompression: On-the-fly or pre-decompress?

---

### 7.2 RUNTIME MEMORY ACCOUNTING

**MANDATORY: Stay within 2GB VRAM ceiling.**

#### 7.2.1 Memory Budget Breakdown

**REQUIRED OUTPUT:**
- Model weights: ~500MB (quantized)
- Activations: Peak during forward pass
- KV cache (if any): SSM state storage
- Working memory: Temporary buffers
- Overhead: Framework, CUDA context
- Total: Must sum to <2GB

#### 7.2.2 Memory Optimization Techniques

**REQUIRED OUTPUT:**
- Gradient checkpointing: Trade compute for memory
- Activation recomputation: Recompute vs store
- In-place operations: Where safe to overwrite
- Memory pooling: Reuse buffers

---

### 7.3 CPU VS GPU ROLE SPLIT

**REQUIRED OUTPUT:**
- GPU: Matrix operations, parallel computation
- CPU: Control flow, I/O, system calls
- Transfer: Minimize CPU-GPU data movement
- Hybrid: What runs where?

---

### 7.4 THROUGHPUT VS LATENCY TRADEOFFS

**REQUIRED OUTPUT:**
- Batch size: 1 for latency, N for throughput
- Streaming: Generate tokens incrementally
- Prefill: Process prompt in parallel
- Decode: Generate autoregressively
- Optimization target: Specify use case (chat, coding, agent)

---

## SECTION 8: BUILD PLAN (PHASED)

**MANDATORY: 10-phase implementation plan.**

### Phase 1: Requirements Breakdown
- Deliverable: Detailed specification document
- Success criteria: All sections reviewed and approved

### Phase 2: Architecture Specification
- Deliverable: Complete architectural diagrams
- Success criteria: Component interfaces defined

### Phase 3: Mathematical Formulation
- Deliverable: All equations, algorithms, proofs
- Success criteria: Mathematical consistency verified

### Phase 4: Systems Design (Rust-Based Runtime)
- Deliverable: Rust implementation of core runtime
- Success criteria: Memory-safe, zero-cost abstractions

### Phase 5: Model Implementation
- Deliverable: KAN + SSM implementation
- Success criteria: Forward/backward pass working

### Phase 6: Verification Pipeline
- Deliverable: Lean/Coq/Z3 integration
- Success criteria: End-to-end proof checking

### Phase 7: Action System
- Deliverable: OS interaction layer
- Success criteria: Can perform basic file/process operations

### Phase 8: Training/Self-Improvement Loop
- Deliverable: Active inference training
- Success criteria: Model improves on self-generated tasks

### Phase 9: Testing + Validation
- Deliverable: Comprehensive test suite
- Success criteria: All tests pass, benchmarks met

### Phase 10: Deployment Constraints
- Deliverable: Deployment guide, resource requirements
- Success criteria: Runs on target hardware

---

## SECTION 9: VALIDATION + TESTING

### 9.1 Explicit Success Criteria

**REQUIRED OUTPUT:**
- Parameter count: ≤3B
- Model size: 400-500MB (quantized)
- Runtime VRAM: ≤2GB
- Memory growth: O(1) verified empirically
- Determinism: Same input → same output (given fixed seed)
- Verifiability: >X% of outputs formally verified (specify X)

### 9.2 Measurable Outputs

**REQUIRED OUTPUT:**
- Benchmarks: Which tasks to evaluate?
- Metrics: Accuracy, speed, memory usage, verification rate
- Baselines: Compare against what?
- Targets: Specific numerical goals

### 9.3 Failure Case Handling

**REQUIRED OUTPUT:**
- Graceful degradation: What happens when limits exceeded?
- Error propagation: How failures are reported
- Recovery: Automatic or manual?
- Logging: What to record for debugging?

### 9.4 Separation of Concerns

**REQUIRED OUTPUT:**
- Solved engineering: Components with known solutions
- Open research: Components requiring novel research
- Risk assessment: Probability of success for each open area
- Fallback: Alternative approaches if research fails

---

## SECTION 10: OUTPUT REQUIREMENTS

### 10.1 Document Structure

Your output must be a single, unified technical document containing:

1. **Executive Summary** (1 page)
2. **System Overview** (architecture diagram + component list)
3. **Detailed Specifications** (all sections above, fully elaborated)
4. **Mathematical Appendix** (all equations, proofs, algorithms)
5. **Implementation Guide** (pseudocode, data structures, APIs)
6. **Validation Protocol** (test plans, benchmarks, success criteria)
7. **Risk Assessment** (technical risks, mitigations)
8. **Timeline** (phase durations, dependencies)

### 10.2 Quality Standards

**MANDATORY:**
- Every claim must have: equation, algorithm, or reference
- No undefined terms: Define all variables, functions, concepts
- No hand-waving: "We could use X" → "We implement X as follows"
- Consistent notation: Same symbol means same thing throughout
- Buildable: Another engineer can implement from your spec
- Verifiable: Success criteria are objectively measurable

### 10.3 Prohibited Content

**DO NOT INCLUDE:**
- Narrative or storytelling
- Marketing language
- Vague aspirations ("we aim to", "we hope to")
- Unsubstantiated claims
- References to future work not in this design
- Fictional or impossible components

---

## SECTION 11: SOLUTIONS TO PREVIOUS LIMITATIONS

### 11.1 GENERAL UNSUPERVISED REASONING VERIFICATION

**Problem:** Cannot verify arbitrary natural language reasoning.

**Solution:** Swarm-based autoformalization curriculum (Section 4.2)

The system progressively learns to encode natural language into Lean through:
1. Starting with simple mathematical statements
2. Gradually increasing complexity through curriculum
3. Using verified Lean equations as building blocks
4. Iterating millions of times in minutes via parallel swarm
5. Eventually encoding complex natural language statements

**Implementation:**
- Level 0-4: Mathematical and logical foundations
- Level 5: Natural language statements
- Level 6: Complex multi-step reasoning
- Each level uses previous levels' verified equations
- Human verification gate ensures correctness
- System learns from feedback and improves over time

---

### 11.2 OPEN-ENDED CREATIVITY

**Problem:** Cannot generate truly novel concepts beyond training distribution.

**Solution:** Swarm exploration + curriculum progression + verified knowledge accumulation

The system achieves bounded creativity through:
1. **Diverse exploration:** Thousands of students try different approaches
2. **Selection pressure:** Teacher selects best solutions
3. **Knowledge accumulation:** Verified solutions become building blocks
4. **Curriculum progression:** Harder problems require novel combinations
5. **Compositional creativity:** New solutions from verified components

**Implementation:**
- Swarm generates diverse hypotheses (N = 100-10000)
- Pruning selects top K solutions
- Verified solutions added to library
- Teacher generates harder problems using library
- System can compose previous solutions in novel ways
- Creativity emerges from composition, not generation ex nihilo

---

### 11.3 HUMAN-LEVEL COMMON SENSE

**Problem:** 3B parameters insufficient for broad world knowledge.

**Solution:** Grounded learning + verified knowledge base + active inference

The system builds common sense through:
1. **Grounded learning:** Knowledge from axioms, not internet text
2. **Verified knowledge:** Every fact is formally verified
3. **Active inference:** System seeks observations to reduce uncertainty
4. **Iterative refinement:** Continuously improves belief state
5. **Compositional reasoning:** Combines verified facts for new insights

**Implementation:**
- Knowledge constructed from:
  - Formal systems (mathematics, logic)
  - Physics-based simulation
  - Code execution environments
  - Verified datasets
- Every fact has proof certificate
- System actively tests hypotheses against environment
- Common sense emerges from verified interactions, not memorization

---

### 11.4 UNBOUNDED SELF-IMPROVEMENT

**Problem:** Will plateau without external data.

**Solution:** Swarm-based self-improvement + curriculum progression + knowledge accumulation

The system achieves continuous improvement through:
1. **Self-generated curriculum:** Teacher creates progressively harder problems
2. **Swarm exploration:** Students generate diverse solutions
3. **Verification gate:** Only verified solutions persist
4. **Knowledge accumulation:** Library grows over time
5. **Compositional improvement:** New solutions build on old ones
6. **Architecture evolution:** Limited structural optimization (within constraints)

**Implementation:**
- Teacher generates problems using verified library
- Students solve problems via swarm
- Solutions verified and added to library
- Teacher generates harder problems
- Process repeats indefinitely
- Improvement bounded by:
  - Compute/memory limits
  - Verification complexity
  - Curriculum progression speed
- But NO plateau from lack of external data

---

## FINAL DELIVERABLE CHECKLIST

Before submitting your design, verify:

- [ ] All 11 sections are complete and detailed
- [ ] All equations use consistent notation
- [ ] All algorithms have pseudocode
- [ ] All components have interfaces defined
- [ ] All success criteria are measurable
- [ ] All constraints are respected
- [ ] No speculative physics or impossible compute
- [ ] Document is internally consistent
- [ ] Document is buildable by another agent
- [ ] Autoformalization pipeline is fully specified
- [ ] Swarm classroom mechanism is detailed
- [ ] Solutions to limitations are incorporated

---

## REMINDER

You are designing a **real system** that must:
1. Fit in 400-500MB on disk
2. Run in 2GB VRAM
3. Use only known methods
4. Be verifiable and deterministic
5. Act on a computer natively
6. Improve itself continuously
7. Encode natural language into formal logic via swarm
8. Operate as execution-first, not text-first
9. Function offline without API dependencies
10. Maintain purpose as invariant objective
11. Detect its own errors automatically
12. Improve through structure, not scale

This is not science fiction. This is engineering. Make it work.
