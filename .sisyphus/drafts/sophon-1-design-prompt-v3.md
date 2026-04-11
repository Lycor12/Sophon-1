# SOPHON-1: META-PROMPT FOR SYSTEM DESIGN (v3)

## EXECUTIVE MANDATE

You are tasked with producing a **complete, buildable, mathematically rigorous system design** for Sophon-1: a compact, computer-native, self-improving AI system operating under strict physical constraints. This is NOT a research proposal. This is an engineering blueprint.

**Your output must be a single, unified technical document** that another autonomous coding agent can use to implement the entire system from first principles. Every claim must be backed by equations, algorithms, or architectural specifications. No hand-waving. No "we could use X." Only "we implement X as follows."

---

## CRITICAL: RESEARCH DIRECTION IS HUMAN-SPECIFIED

**The implementing AI MUST NOT invent novel research directions.** All research frontiers are specified in this document. The AI's role is to:

1. **Implement** the architectures described herein
2. **Optimize** within the defined parameter space
3. **Integrate** components as specified
4. **NOT** propose new methods beyond what is documented

If the implementing AI encounters a gap in this specification, it must:
- Request clarification from human supervisors
- Use the most conservative interpretation
- Document the assumption explicitly
- NOT invent novel solutions without human approval

This prevents hallucination of unproven methods while allowing the system to push into research frontiers under human guidance.

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
| **Target Capability** | Narrow AGI in coding/math/science | Superhuman in specific domains, not general-purpose |

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

## SECTION 0: HUMAN-SPECIFIED RESEARCH ROADMAP

**This section defines ALL research frontiers. The implementing AI must follow these directions exactly.**

### 0.1 KAN ARCHITECTURE RESEARCH DIRECTIONS

**Human-specified approach (DO NOT DEVIATE):**

The implementing AI must implement KAN using the following research directions:

#### 0.1.1 Adaptive Spline Bases

**Research direction:** Use B-splines with adaptive knot placement.

**Implementation requirements:**
- Start with uniform knots
- Implement gradient-based knot optimization
- Knot positions are learnable parameters
- Regularization: Penalize knot movement to prevent instability

**Mathematical formulation:**
```
φ(x) = Σᵢ cᵢ · Bᵢ,ₖ(x; t₀, t₁, ..., tₙ)

Where:
- t₀, t₁, ..., tₙ are learnable knot positions
- Gradient: ∂φ/∂tⱼ computed via chain rule through basis functions
- Constraint: t₀ < t₁ < ... < tₙ (enforced via reparameterization)
```

**DO NOT:**
- Invent new basis functions
- Use unproven functional forms
- Speculate on alternative representations

#### 0.1.2 Function Composition Strategy

**Research direction:** Compose functions across layers via direct summation.

**Implementation requirements:**
- Each layer: yⱼ = Σᵢ φᵢⱼ(xᵢ)
- No complex composition operators
- Simplicity over expressivity

**DO NOT:**
- Invent new composition operators
- Use unproven functional compositions

---

### 0.2 SSM ARCHITECTURE RESEARCH DIRECTIONS

**Human-specified approach (DO NOT DEVIATE):**

#### 0.2.1 State Space Parameterization

**Research direction:** Use diagonal-plus-low-rank parameterization for stability.

**Implementation requirements:**
- A = diag(λ₁, ..., λₙ) + U·Vᵀ where U, V are low-rank
- Constraint: Re(λᵢ) < 0 for stability
- Low-rank component: rank r << n

**Mathematical formulation:**
```
A = -exp(S) + U·Vᵀ

Where:
- S is learned (ensures negative eigenvalues for diagonal)
- U ∈ ℝ^(n×r), V ∈ ℝ^(n×r) are learned low-rank matrices
- r = n/8 or n/16 (specify exact ratio)
```

**DO NOT:**
- Use unproven parameterizations
- Invent new stability mechanisms

#### 0.2.2 Discretization Method

**Research direction:** Use Zero-Order Hold (ZOH) discretization.

**Implementation requirements:**
- Ā = exp(A·Δt)
- B̄ = A⁻¹·(Ā - I)·B
- Δt is learned per layer

**DO NOT:**
- Invent new discretization schemes
- Use unproven numerical methods

---

### 0.3 KAN + SSM INTEGRATION RESEARCH

**Human-specified approach (DO NOT DEVIATE):**

#### 0.3.1 Block Architecture

**Research direction:** Interleaved blocks with residual connections.

**Implementation requirements:**
```
For each layer l:
  x' = KAN_l(x)
  x'' = SSM_l(x')
  output = x + x''  (residual)
```

**DO NOT:**
- Invent new integration patterns
- Use unproven hybrid architectures

---

### 0.4 AUTOFORMALIZATION RESEARCH DIRECTIONS

**Human-specified approach (DO NOT DEVIATE):**

#### 0.4.1 Swarm-Based Translation

**Research direction:** Use swarm classroom for all autoformalization.

**Implementation requirements:**
- Teacher generates problems from verified library
- Students (N = 1000-10000) generate solutions in parallel
- Top K solutions selected via verification
- Translation to Lean via separate swarm
- Human verification gate for accepted equations

**DO NOT:**
- Use single-model translation
- Invent new autoformalization methods
- Skip the human verification gate

#### 0.4.2 Curriculum Progression

**Research direction:** Level-based curriculum with strict thresholds.

**Implementation requirements:**
- Level 0-6 as defined in Section 4.2.2
- Progress only when threshold met
- Use previous levels' equations for harder problems

**DO NOT:**
- Invent new curriculum structures
- Skip levels or lower thresholds

---

### 0.5 SELF-IMPROVEMENT RESEARCH DIRECTIONS

**Human-specified approach (DO NOT DEVIATE):**

#### 0.5.1 Active Inference Framework

**Research direction:** Free energy minimization with variational approximation.

**Implementation requirements:**
- Free energy: F = E_q[log q(s) - log p(o,s)]
- Variational approximation: q(s) = Gaussian with diagonal covariance
- Gradient descent on F

**DO NOT:**
- Invent new objective functions
- Use unproven learning rules

#### 0.5.2 Knowledge Accumulation

**Research direction:** Verified Lean equations as building blocks.

**Implementation requirements:**
- All knowledge stored as verified Lean equations
- New problems use existing equations
- No unverified knowledge in library

**DO NOT:**
- Store unverified information
- Invent new knowledge representation schemes

---

### 0.6 QUANTIZATION RESEARCH DIRECTIONS

**Human-specified approach (DO NOT DEVIATE):**

#### 0.6.1 Ternary Quantization

**Research direction:** 1.58-bit quantization for all linear operations.

**Implementation requirements:**
- Values: {-1, 0, +1}
- Packing: 32 weights per 64-bit word
- Straight-through estimator for gradients

**DO NOT:**
- Invent new quantization schemes
- Use unproven compression methods

---

## SECTION 1: MODEL CORE ARCHITECTURE

### 1.1 KOLMOGOROV-ARNOLD NETWORK (KAN) CORE

**MANDATORY: Replace ALL traditional linear layers with learnable edge functions.**

#### 1.1.1 Functional Basis Specification

**REQUIRED: Use B-Splines with adaptive knots (per Section 0.1.1)**

**Mathematical form:**
```
φ(x) = Σᵢ cᵢ · Bᵢ,ₖ(x; t₀, t₁, ..., tₙ)

Where:
- k = 3 (cubic splines)
- t₀, t₁, ..., tₙ are learnable knot positions
- cᵢ are learnable coefficients
- n = 5-10 knots per edge (specify exact count)
```

**REQUIRED OUTPUT:**
- Complete B-spline implementation with adaptive knots
- Gradient computation for both coefficients and knots
- Knot ordering constraint enforcement
- Parameter count per edge function

#### 1.1.2 Forward Pass Formulation

Define the complete forward pass equation:

```
For layer l with nₗ inputs and nₗ₊₁ outputs:

yⱼ⁽ˡ⁺¹⁾ = Σᵢ₌₁ⁿˡ φᵢⱼ⁽ˡ⁾(xᵢ⁽ˡ⁾) + bⱼ⁽ˡ⁾

Where:
- φᵢⱼ⁽ˡ⁾: Learnable B-spline function on edge (i,j) at layer l
- xᵢ⁽ˡ⁾: i-th input to layer l
- yⱼ⁽ˡ⁺¹⁾: j-th output from layer l
- bⱼ⁽ˡ⁾: Optional bias
```

**REQUIRED OUTPUT:**
- Complete forward pass pseudocode with O-notation complexity
- Memory access pattern (cache-friendly ordering)
- Numerical stability considerations
- Batch processing implementation

#### 1.1.3 Gradient Propagation Through Function Space

Define how gradients flow:

```
∂L/∂cᵢⱼₖ = ∂L/∂yⱼ · ∂yⱼ/∂φᵢⱼ · ∂φᵢⱼ/∂cᵢⱼₖ
∂L/∂tⱼ = ∂L/∂y · ∂y/∂φ · ∂φ/∂tⱼ

Where:
- ∂φ/∂c: Derivative of B-spline w.r.t. coefficients
- ∂φ/∂t: Derivative of B-spline w.r.t. knot positions
```

**REQUIRED OUTPUT:**
- Backpropagation algorithm for KAN layers
- Gradient computation for both coefficients and knots
- Gradient checkpointing strategy
- Second-order methods feasibility

#### 1.1.4 Function Sparsity and Pruning

Define sparsity enforcement:

**REQUIRED OUTPUT:**
- Sparsity metric: L1 norm of coefficients
- Pruning threshold: λ (specify schedule)
- Pruning schedule: Start after N epochs, prune every M steps
- Recovery mechanism: Re-grow edges if needed
- Final sparsity target: 70-80% edges active

**Mathematical form:**
```
Regularized loss: L_total = L_task + λ · Σᵢⱼ ||cᵢⱼ||₁

Pruning condition: if ||cᵢⱼ|| < ε then φᵢⱼ ← 0
```

---

### 1.2 STATE SPACE MEMORY SYSTEM (SSM)

**MANDATORY: Replace attention mechanisms with continuous-time state evolution.**

#### 1.2.1 Formal State Update Equation

**REQUIRED: Use diagonal-plus-low-rank parameterization (per Section 0.2.1)**

**Continuous form (ODE):**
```
ḣ(t) = A·h(t) + B·x(t)
y(t) = C·h(t) + D·x(t)

Where:
A = -exp(S) + U·Vᵀ
```

**Discrete approximation (ZOH):**
```
hₖ = Ā·hₖ₋₁ + B̄·xₖ
yₖ = C·hₖ + D·xₖ

Where:
Ā = exp(A·Δt)
B̄ = A⁻¹·(Ā - I)·B
Δt is learned per layer
```

**REQUIRED OUTPUT:**
- Explicit discrete update equations
- Matrix exponential computation (Taylor series or Pade approximation)
- Numerical stability proof
- Δt initialization and learning

#### 1.2.2 Hidden State Dimensionality

**REQUIRED OUTPUT:**
- State dimension N: 128 (specify exact value)
- Input dimension D: 256 (specify exact value)
- Output dimension P: 256 (specify exact value)
- Low-rank r: N/8 = 16
- Memory per sequence element: O(N) = O(128)
- Total memory: O(N) constant across sequence length

#### 1.2.3 Input Injection Mechanism

Define how inputs enter the state:

```
xₖ = W_in · embeddingₖ + b_in
xₖ' = LayerNorm(xₖ)
hₖ = SSM_update(hₖ₋₁, xₖ')
```

**REQUIRED OUTPUT:**
- Input projection architecture
- LayerNorm placement
- Skip connections

#### 1.2.4 Stability Constraints

**MANDATORY: Define stability guarantees.**

**REQUIRED OUTPUT:**
- Eigenvalue constraint: Re(λᵢ(A)) < 0 enforced via -exp(S) parameterization
- Low-rank perturbation stability analysis
- Long-range dependency: Information retention for T = 10,000+ steps

**Mathematical proof sketch:**
```
For A = -exp(S) + U·Vᵀ:
- Diagonal part: eigenvalues = -exp(Sᵢᵢ) < 0
- Low-rank perturbation: ||U·Vᵀ|| ≤ ||U||·||V||
- Stability condition: ||U·Vᵀ|| < min|exp(Sᵢᵢ)|
```

---

### 1.3 HYBRID KAN + SSM INTEGRATION

**MANDATORY: Use interleaved blocks (per Section 0.3.1)**

#### 1.3.1 Block Structure

**REQUIRED OUTPUT:**
```
Block architecture:
  Input x
  → LayerNorm
  → KAN layer
  → LayerNorm
  → SSM layer
  → Residual connection: output = x + SSM_output
```

**REQUIRED OUTPUT:**
- Complete block definition
- LayerNorm placement
- Residual connection implementation
- Number of blocks: 12-24 (specify exact count)

#### 1.3.2 Data Flow Specification

**REQUIRED OUTPUT:**
- Tensor shapes at each stage
- Residual connections: Around each block
- Normalization: LayerNorm before KAN and SSM
- Activation functions: None (KAN provides nonlinearity)

#### 1.3.3 Latency and Compute Analysis

**REQUIRED OUTPUT:**
- Per-block FLOPs: KAN (O(n²·k)) + SSM (O(n·d))
- Memory bandwidth: Bytes per parameter accessed
- Parallelization: KAN edges parallel, SSM sequential
- Critical path: SSM recurrence

---

## SECTION 2: REPRESENTATION LAYER

### 2.1 TOKEN / INPUT ENCODING

**MANDATORY: Define encoding strategy respecting memory budget.**

#### 2.1.1 Tokenization Strategy

**REQUIRED: Use byte-level tokenization**

- Vocabulary: 256 (raw bytes)
- Advantage: No OOV, universal, minimal memory
- Disadvantage: Longer sequences (acceptable for O(1) memory)

**REQUIRED OUTPUT:**
- Vocabulary size V = 256
- Embedding matrix dimensions: 256 × d_model
- Memory cost: 256 × d_model × bytes_per_param

#### 2.1.2 Embedding Method

**REQUIRED OUTPUT:**
- Standard embedding: E ∈ ℝ^(256×d)
- d_model = 256 (specify exact value)
- No factorization needed (vocabulary is tiny)

---

### 2.2 HYPERDIMENSIONAL / VECTOR SYMBOLIC REPRESENTATIONS

**MANDATORY: Enable symbolic manipulation in neural space.**

#### 2.2.1 Compositional Binding

**REQUIRED: Use circular convolution**

```
a ⊗ b = IFFT(FFT(a) ⊙ FFT(b))
```

**REQUIRED OUTPUT:**
- Binding operation implementation
- Dimensionality: d = 2048 (specify exact value)
- Unbinding: a ⊗⁻¹ c ≈ b where c = a ⊗ b
- Cleanup memory: Nearest neighbor lookup in codebook

#### 2.2.2 Variable Binding for Symbolic Reasoning

**REQUIRED OUTPUT:**
- Role-filler binding: bind(role, filler)
- Example: bind("subject", "cat") ⊗ bind("action", "chase")
- Cleanup memory: How to retrieve from noisy representations
- Stack representation: Position encoding via binding

---

### 2.3 LATENT WORLD MODEL

**MANDATORY: Internal representation of world state.**

#### 2.3.1 Belief State vs Observation

**REQUIRED OUTPUT:**
- Observation oₜ: System state at time t (process table, files, network)
- Belief state bₜ: Internal representation P(sₜ | o₁:ₜ)
- Separation: Belief updated via active inference

#### 2.3.2 Active Inference Alignment

**REQUIRED OUTPUT:**
- Generative model: p(o, s) = p(o|s)·p(s)
- Recognition model: q(s|o) = Gaussian
- Free energy: F = E_q[log q(s) - log p(o,s)]
- Perception: Minimize F w.r.t. q
- Action: Minimize F w.r.t. a

---

## SECTION 3: LEARNING DYNAMICS

### 3.1 ACTIVE INFERENCE OBJECTIVE

**MANDATORY: Use free energy minimization (per Section 0.5.1)**

#### 3.1.1 Variational Free Energy

Define:
```
F = E_q(s)[log q(s) - log p(o,s)]
  = D_KL[q(s) || p(s)] - E_q[log p(o|s)]
```

**REQUIRED OUTPUT:**
- Explicit free energy computation
- Variational approximation: q(s) = N(μ, σ²I)
- KL divergence: Analytical for Gaussian
- Gradient: ∂F/∂μ, ∂F/∂σ

#### 3.1.2 Prediction Error

**REQUIRED OUTPUT:**
- Sensory prediction: ŏ = g(s)
- Prediction error: ε = o - ŏ
- Precision weighting: π · ε²

#### 3.1.3 Complexity Penalty

**REQUIRED OUTPUT:**
- Prior p(s) = N(0, I)
- Posterior q(s) = N(μ, σ²I)
- KL divergence: ½(μ² + σ² - log(σ²) - 1)

---

### 3.2 SELF-IMPROVEMENT LOOP

**MANDATORY: Internal iteration without external data dependency.**

#### 3.2.1 Internal Rollout Mechanism

**REQUIRED OUTPUT:**
- Hypothesis generation: Sample from q(s)
- Mental simulation: Forward model predicts outcomes
- Verification: Check against verified Lean library
- Update: Adjust parameters via gradient descent

#### 3.2.2 Hypothesis → Verification → Update Cycle

Define the loop:
```
For iteration i:
  1. Generate hypothesis hᵢ ~ q(·|current_state)
  2. Simulate outcome õᵢ = forward_model(hᵢ)
  3. Verify: Is õᵢ consistent with Lean library?
  4. If verified: Update parameters θ ← θ - α·∂L/∂θ
  5. If failed: Generate counter-example, update with negative gradient
```

**REQUIRED OUTPUT:**
- Hypothesis sampling strategy
- Verification criteria
- Failure handling
- Convergence criteria

#### 3.2.3 Learning Without Massive Datasets

**REQUIRED OUTPUT:**
- Synthetic data generation: Self-generated problems
- Curriculum: Progressive difficulty
- Replay buffer: Store verified solutions
- Exploration: Diverse hypothesis generation

---

### 3.3 PARAMETER-EFFICIENT ADAPTATION

**MANDATORY: Enable rapid specialization without full retraining.**

#### 3.3.1 LoRA-Style Low-Rank Adapters

Define:
```
W_adapted = W_base + B·A

Where:
- W_base: Frozen base weights
- B ∈ ℝ^(d×r), A ∈ ℝ^(r×k)
- r = 16 (specify exact value)
```

**REQUIRED OUTPUT:**
- Adapter placement: On KAN coefficient matrices and SSM A/B/C matrices
- Initialization: A = random normal, B = zero
- Scaling: α/r with α = 32
- Memory: Only store B and A

#### 3.3.2 Adapter Attachment Points

**REQUIRED OUTPUT:**
- KAN edges: LoRA on coefficient matrices cᵢⱼ
- SSM states: LoRA on A, B, C matrices
- Implementation details

#### 3.3.3 Merging and Pruning Strategy

**REQUIRED OUTPUT:**
- Merging: After task completion, merge adapters into base
- Pruning: Remove adapters with ||B·A|| < ε
- Selection: Keep adapters with best task performance

---

## SECTION 4: ZERO-HALLUCINATION SYSTEM

### 4.1 NEUROSYMBOLIC VERIFICATION LAYER

**MANDATORY: Use Lean 4 for formal verification**

#### 4.1.1 Formal System Choice

**REQUIRED: Lean 4**

- Advantages: Large math library, active community, fast compilation
- Integration: Export proofs to Lean, check with lake build

**REQUIRED OUTPUT:**
- Lean 4 integration architecture
- Interface: Neural output → Lean syntax
- Proof format: Lean 4 proof terms

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
│  │ N Student Agents (N = 1000-10000):                        │   │
│  │ - Each student receives problem P                         │   │
│  │ - Students generate diverse solution candidates           │   │
│  │ - Different hyperparameters, random seeds, approaches     │   │
│  │ - Parallel execution (GPU batch inference)                │   │
│  │ - Time limit: T_solve = 50ms per student                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           ↓                                      │
│  Stage 3: PRUNING AND SELECTION                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Teacher Agent: Evaluates all N solutions                  │   │
│  │ - Score each solution: correctness, elegance, speed       │   │
│  │ - Prune bottom 95% of students                            │   │
│  │ - Keep top K = 50 solutions                               │   │
│  │ - Terminate pruned students (free memory)                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           ↓                                      │
│  Stage 4: LEAN TRANSLATION SWARM                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ For each of top K solutions:                              │   │
│  │ - Spawn M = 100 translation students                      │   │
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
│  │ - Timeout: T_verify = 5s per translation                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           ↓                                      │
│  Stage 6: HUMAN VERIFICATION GATE                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Human Reviewer (MANDATORY for training phase):            │   │
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
  - Iterations: 10,000-100,000

Level 1: Algebraic Identities
  - Problems: "(a + b)² = a² + 2ab + b²"
  - Lean encoding: Ring tactic usage
  - Success threshold: 90% verification rate
  - Iterations: 50,000-500,000

Level 2: Logical Propositions
  - Problems: "P ∧ Q → P", "P ∨ ¬P"
  - Lean encoding: Propositional logic
  - Success threshold: 85% verification rate
  - Iterations: 100,000-1,000,000

Level 3: First-Order Logic
  - Problems: "∀x. P(x) → Q(x), P(a) ⊢ Q(a)"
  - Lean encoding: Quantifier reasoning
  - Success threshold: 80% verification rate
  - Iterations: 500,000-5,000,000

Level 4: Code Correctness
  - Problems: "sort(l) is sorted", "reverse(reverse(l)) = l"
  - Lean encoding: Program verification
  - Success threshold: 70% verification rate
  - Iterations: 1,000,000-10,000,000

Level 5: Natural Language Statements
  - Problems: "All humans are mortal. Socrates is human. Therefore, Socrates is mortal."
  - Lean encoding: NL → formal translation
  - Success threshold: 50% verification rate (initial target)
  - Iterations: 5,000,000-50,000,000

Level 6: Complex Reasoning
  - Problems: Multi-step reasoning chains in coding/math/science
  - Lean encoding: Composed proofs using previous levels
  - Success threshold: 40% verification rate (initial target)
  - Iterations: 10,000,000-100,000,000
```

**Progression rule:** Teacher only advances to Level N+1 when Level N achieves success threshold.

**Speed:** Each iteration takes 200ms-2s. At 1 iteration/second, Level 6 requires ~100M seconds = ~3 years of continuous training. With parallelization (1000 GPUs), this reduces to ~1 day.

#### 4.2.3 Speed and Efficiency Requirements

**REQUIRED OUTPUT:**

| Stage | Time Budget | Parallelism |
|-------|-------------|-------------|
| Problem Generation | 5 ms | Single teacher |
| Student Exploration | 50 ms | N=10000 students in parallel |
| Pruning | 5 ms | Single teacher |
| Lean Translation | 50 ms | M=100 students per solution |
| Lean Compilation | 500 ms | Lean compiler |
| Human Review | 1-10s (optional after training) | Human-in-loop |
| **Total (automated)** | **~600 ms** | Per problem |

**Optimization strategies:**
- Batch inference: Process N students in single GPU batch
- Early termination: Stop students that exceed time budget
- Caching: Reuse compiled Lean libraries
- Incremental compilation: Only compile new equations

#### 4.2.4 Encoding Human Language

**MANDATORY: Define how the system learns to encode natural language into Lean.**

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

#### 4.2.5 Swarm Hyperparameters

**REQUIRED OUTPUT:**

```
Student Swarm Configuration:
- N_students: 10,000 (fixed)
- Diversity mechanisms:
  - Different random seeds (10,000 seeds)
  - Different LoRA adapters (100 adapters × 100 seeds)
  - Temperature: 0.5-1.5 (sampled)
- Resource limits per student:
  - Max tokens: 500
  - Max time: 50ms
  - Max memory: 5MB

Translation Swarm Configuration:
- M_translators: 100 per solution
- Translation strategies:
  - Direct mapping (learned)
  - Rule-based translation
  - Template-based (for known patterns)

Pruning Strategy:
- Selection: Top K = 50 by verification score
- Termination: Kill pruned students immediately
- Memory management: Reuse student slots
```

#### 4.2.6 Feedback Loop and Learning

**REQUIRED OUTPUT:**

```
Learning Mechanisms:

1. Successful verification:
   - Reinforce student weights that produced correct solution
   - Add Lean equation to verified library
   - Update translation model with successful NL→Lean pair
   - Teacher generates harder problem

2. Failed verification:
   - Analyze error type (syntax, type, proof failed)
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
   - Adaptive difficulty

5. Knowledge accumulation:
   - Verified Lean equations become building blocks
   - Harder problems use previous equations
   - Library grows over time
   - System becomes more capable
```

---

### 4.3 OUTPUT CONSTRAINT MECHANISM

**MANDATORY: No unverified outputs permitted.**

#### 4.3.1 Fallback Behavior

**REQUIRED OUTPUT:**
- When proof fails: Output "I cannot verify this" + uncertainty quantification
- Provide best attempt with "UNVERIFIED" tag
- User-configurable strictness level

#### 4.3.2 Iterative Refinement Loop

**REQUIRED OUTPUT:**
- When verification fails:
  1. Analyze failure
  2. Generate alternative (swarm)
  3. Re-verify
  4. Repeat up to N = 10 attempts
- Timeout: 10 seconds maximum

---

### 4.4 TOOL-ASSISTED TRUTH ACQUISITION

**MANDATORY: External data treated as noisy observation.**

#### 4.4.1 Noisy Observation Model

**REQUIRED OUTPUT:**
- External tool output t: Observation with uncertainty
- Precision weighting: π_t based on tool reliability
- Integration: Update belief via precision-weighted error

#### 4.4.2 Verification Before Integration

**REQUIRED OUTPUT:**
- Cross-validation: Multiple tools for critical facts
- Consensus: k of n tools must agree
- Temporal validation: Check consistency
- Source tracking: Maintain provenance

---

## SECTION 5: COMPUTER-NATIVE ACTION SYSTEM

### 5.1 DIRECT SYSTEM INTERACTION

**MANDATORY: Operate via OS primitives, not APIs.**

#### 5.1.1 OS Primitive Operations

**REQUIRED OUTPUT:**
- System calls: Direct syscall via libc wrapper
- Privilege model: User-level, no root required
- Sandboxing: Resource limits via cgroups (Linux) or Job Objects (Windows)

#### 5.1.2 Filesystem Operations

**REQUIRED OUTPUT:**
- Operations: read, write, append, delete, list, stat
- Path handling: Absolute paths, symlink resolution
- Atomicity: Rename for atomic updates
- Quotas: Max 10GB disk usage

#### 5.1.3 Process Control

**REQUIRED OUTPUT:**
- Spawn: fork/exec (Linux) or CreateProcess (Windows)
- Monitor: waitpid or WaitForSingleObject
- Signal: kill or TerminateProcess
- Resource limits: CPU time, memory, file descriptors

---

### 5.2 VISION-LANGUAGE-ACTION LOOP

**MANDATORY: Perceive screen, decide action, execute.**

#### 5.2.1 Screen Perception Pipeline

**REQUIRED OUTPUT:**
- Capture: X11 (Linux) or Win32 API (Windows)
- Resolution: Downsample to 256×256
- Encoding: Raw RGB pixels
- Frequency: On-demand (event-driven)

#### 5.2.2 Action Execution Cycle

**REQUIRED OUTPUT:**
- Action space:
  - Mouse: move, click, scroll
  - Keyboard: key press, hotkey
  - System: execute command
- Latency: <100ms target
- Safety: Confirmation for destructive actions

#### 5.2.3 Latency Constraints

**REQUIRED OUTPUT:**
- Perception: 10ms
- Inference: 50ms
- Action: 10ms
- Total: <100ms

---

### 5.3 BYTE-LEVEL OR LOW-LEVEL INTERFACE

**MANDATORY: Represent system state at byte/process level.**

#### 5.3.1 System State Representation

**REQUIRED OUTPUT:**
- Process table: PID, state, memory, CPU
- File descriptors: Open files, sockets
- Memory map: Heap, stack, mapped regions
- Network: Connections, ports

#### 5.3.2 Byte-Level Manipulation

**REQUIRED OUTPUT:**
- Binary data: Read/write byte sequences
- Parsing: JSON, protobuf, custom formats
- Construction: Build valid messages
- Validation: Checksum, length, format

---

## SECTION 6: SWARM + SELF-PROGENY SYSTEM

### 6.1 MULTI-AGENT INSTANTIATION

**MANDATORY: Lightweight copies via adapters.**

#### 6.1.1 Agent Cloning

**REQUIRED OUTPUT:**
- Base model: Shared read-only weights (~500MB)
- Specialization: Unique LoRA adapters (~10MB per agent)
- Memory: 500MB + 10MB × N agents
- Communication: Shared memory for read-only, message passing for private state

#### 6.1.2 Isolation Boundaries

**REQUIRED OUTPUT:**
- Process isolation: Separate threads (shared memory) or processes (isolated)
- Resource isolation: CPU/memory quotas
- Failure isolation: Catch exceptions, don't crash parent
- Security isolation: No access to other agents' private state

---

### 6.2 CLASSROOM TRAINING LOOP

**MANDATORY: Multi-agent learning with roles.**

#### 6.2.1 Role Definitions

**REQUIRED OUTPUT:**
- Teacher agent:
  - Responsibilities: Problem generation, verification, selection
  - Capabilities: Lean compiler access, curriculum control
  - Authority: Promote/demote students

- Student agents:
  - Responsibilities: Solution generation, translation
  - Diversity: Different seeds, adapters, temperatures
  - Lifecycle: Created, evaluated, merged, pruned

#### 6.2.2 Shared Knowledge Base

**REQUIRED OUTPUT:**
- Storage: Verified Lean equations
- Access: Read-only for students, read-write for teacher
- Consistency: Append-only, no deletion

#### 6.2.3 Competition and Merging

**REQUIRED OUTPUT:**
- Competition: Evaluated on problem solving
- Selection: Top K = 50 survive
- Merging: EMA of weights
  - θ_ensemble = 0.9·θ_ensemble + 0.1·θ_winner

---

### 6.3 SELF-EVOLUTION CONSTRAINTS

**MANDATORY: Prevent divergence from core objective.**

#### 6.3.1 Stability Mechanisms

**REQUIRED OUTPUT:**
- Objective anchoring: ||θ - θ₀|| < ε
- ε = 0.1 (max 10% parameter drift)
- Validation: Every 1000 iterations
- Rollback: If performance drops >5%, revert

#### 6.3.2 Alignment Verification

**REQUIRED OUTPUT:**
- Alignment metric: Performance on core tasks
- Monitoring: Continuous
- Intervention: Automatic rollback if degraded

---

## SECTION 7: MEMORY + EFFICIENCY

### 7.1 QUANTIZATION STRATEGY

**MANDATORY: Use ternary quantization (per Section 0.6.1)**

#### 7.1.1 Quantization Method

**REQUIRED: 1.58-bit (Ternary)**

- Values: {-1, 0, +1}
- Packing: 32 weights per 64-bit word
- Implementation: Bitwise operations

**REQUIRED OUTPUT:**
- Quantization for all linear operations
- Scale factor per tensor
- Straight-through estimator for gradients

#### 7.1.2 Weight Packing and Loading

**REQUIRED OUTPUT:**
- On-disk format: Custom binary (packed ternary)
- Loading: Memory-mapped
- Decompression: On-the-fly during inference

---

### 7.2 RUNTIME MEMORY ACCOUNTING

**MANDATORY: Stay within 2GB VRAM ceiling.**

#### 7.2.1 Memory Budget Breakdown

**REQUIRED OUTPUT:**
- Model weights: ~500MB (ternary packed)
- Activations: ~200MB (peak)
- SSM state: ~50MB
- Working memory: ~100MB
- Lean compiler: ~500MB
- Overhead: ~150MB
- Total: ~1.5GB (under 2GB limit)

#### 7.2.2 Memory Optimization Techniques

**REQUIRED OUTPUT:**
- Gradient checkpointing: All activations
- Activation recomputation: Recompute during backward
- In-place operations: Where safe
- Memory pooling: Reuse buffers

---

### 7.3 CPU VS GPU ROLE SPLIT

**REQUIRED OUTPUT:**
- GPU: KAN forward/backward, SSM forward/backward
- CPU: Lean compilation, OS operations, orchestration
- Transfer: Minimize (keep model on GPU)
- Hybrid: GPU for inference, CPU for verification

---

### 7.4 THROUGHPUT VS LATENCY TRADEOFFS

**REQUIRED OUTPUT:**
- Batch size: 1 for latency, 32 for throughput
- Streaming: Incremental generation
- Prefill: Parallel prompt processing
- Decode: Autoregressive
- Optimization: Latency for agent mode, throughput for batch mode

---

## SECTION 8: BUILD PLAN (PHASED)

**MANDATORY: 10-phase implementation plan.**

### Phase 1: Requirements Breakdown (Week 1-2)
- Deliverable: Detailed specification document
- Success criteria: All sections reviewed and approved

### Phase 2: Architecture Specification (Week 3-4)
- Deliverable: Complete architectural diagrams
- Success criteria: Component interfaces defined

### Phase 3: Mathematical Formulation (Week 5-6)
- Deliverable: All equations, algorithms, proofs
- Success criteria: Mathematical consistency verified

### Phase 4: Systems Design (Rust Runtime) (Week 7-10)
- Deliverable: Rust implementation of core runtime
- Success criteria: Memory-safe, zero-cost abstractions

### Phase 5: Model Implementation (Week 11-16)
- Deliverable: KAN + SSM implementation
- Success criteria: Forward/backward pass working

### Phase 6: Verification Pipeline (Week 17-20)
- Deliverable: Lean 4 integration
- Success criteria: End-to-end proof checking

### Phase 7: Action System (Week 21-24)
- Deliverable: OS interaction layer
- Success criteria: Basic file/process operations

### Phase 8: Swarm Training Loop (Week 25-32)
- Deliverable: Classroom training implementation
- Success criteria: Curriculum progression working

### Phase 9: Testing + Validation (Week 33-40)
- Deliverable: Comprehensive test suite
- Success criteria: All tests pass, benchmarks met

### Phase 10: Deployment (Week 41-48)
- Deliverable: Deployment guide, resource requirements
- Success criteria: Runs on target hardware

---

## SECTION 9: VALIDATION + TESTING

### 9.1 Explicit Success Criteria

**REQUIRED OUTPUT:**
- Parameter count: ≤3B
- Model size: 400-500MB (ternary packed)
- Runtime VRAM: ≤2GB
- Memory growth: O(1) verified empirically
- Determinism: Same input → same output (fixed seed)
- Verifiability: >80% of outputs formally verified (after training)

### 9.2 Measurable Outputs

**REQUIRED OUTPUT:**
- Benchmarks:
  - Math: IMO problems, theorem proving
  - Code: HumanEval, MBPP, verified code generation
  - Science: Physics problems, chemistry equations
- Metrics: Accuracy, verification rate, speed, memory
- Baselines: GPT-4, Claude 3, AlphaProof
- Targets: Superhuman in math, competitive in code

### 9.3 Failure Case Handling

**REQUIRED OUTPUT:**
- Graceful degradation: Fall back to unverified output with warning
- Error propagation: Structured error messages
- Recovery: Automatic retry with different approach
- Logging: Full trace for debugging

### 9.4 Separation of Concerns

**REQUIRED OUTPUT:**
- Solved engineering: KAN, SSM, quantization, OS operations
- Research implementation: Autoformalization swarm, curriculum
- Risk assessment: Medium risk in autoformalization, low risk elsewhere
- Fallback: Human-in-loop verification if autoformalization fails

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
- Novel research directions not specified in Section 0

---

## FINAL DELIVERABLE CHECKLIST

Before submitting your design, verify:

- [ ] All 10 sections are complete and detailed
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
- [ ] Research directions follow Section 0 exactly
- [ ] No novel methods invented beyond Section 0

---

## REMINDER

You are designing a **real system** that must:
1. Fit in 400-500MB on disk
2. Run in 2GB VRAM
3. Follow human-specified research directions (Section 0)
4. Be verifiable and deterministic
5. Act on a computer natively
6. Improve itself continuously via swarm classroom
7. Encode natural language into formal logic via swarm
8. Operate as execution-first, not text-first
9. Function offline without API dependencies
10. Maintain purpose as invariant objective
11. Detect its own errors automatically
12. Improve through structure, not scale
13. Achieve narrow AGI in coding/math/science domains

**This is not science fiction. This is engineering. Make it work.**
