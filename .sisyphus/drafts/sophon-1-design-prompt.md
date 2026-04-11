# SOPHON-1: META-PROMPT FOR SYSTEM DESIGN

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
- Special tokens: [BOS], [EOS], [PAD], [MASK]

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

#### 4.1.2 Verification Pipeline

Define 4-stage pipeline:

```
Stage 1: Neural Hypothesis Generation
  - Input: Query/context
  - Output: Candidate answer + confidence
  - Mechanism: Forward pass through KAN+SSM

Stage 2: Translation to Formal Logic
  - Input: Neural output
  - Output: Formal statement in target logic
  - Mechanism: Learned translator or rule-based

Stage 3: Proof Checking
  - Input: Formal statement
  - Output: PROVED / FAILED / TIMEOUT
  - Mechanism: Invoke theorem prover

Stage 4: Acceptance/Rejection
  - If PROVED: Emit output with proof certificate
  - If FAILED: Mark as uncertain, trigger refinement
  - If TIMEOUT: Queue for human review or retry with hints
```

**REQUIRED OUTPUT:**
- Complete pipeline architecture
- Latency budget per stage
- Failure modes and handling

---

### 4.2 OUTPUT CONSTRAINT MECHANISM

**MANDATORY: No unverified outputs permitted.**

#### 4.2.1 Fallback Behavior

**REQUIRED OUTPUT:**
- When proof fails: What does the system output?
  - Option A: "I don't know" + uncertainty quantification
  - Option B: Request clarification
  - Option C: Provide answer with explicit "UNVERIFIED" tag
- User-configurable: Allow override for specific contexts?

#### 4.2.2 Iterative Refinement Loop

**REQUIRED OUTPUT:**
- When verification fails, how to refine:
  1. Analyze failure: What part of the proof broke?
  2. Generate alternative: Sample new hypothesis
  3. Re-verify: Check new candidate
  4. Repeat: Up to N attempts (specify N)
- Timeout: Maximum refinement iterations before giving up

---

### 4.3 TOOL-ASSISTED TRUTH ACQUISITION

**MANDATORY: External data treated as noisy observation.**

#### 4.3.1 Noisy Observation Model

**REQUIRED OUTPUT:**
- External tool output t: Treat as observation with uncertainty
- Precision weighting: π_t (reliability of tool)
- Integration: Update belief state using precision-weighted error

#### 4.3.2 Verification Before Integration

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

---

## REMINDER

You are designing a **real system** that must:
1. Fit in 400-500MB on disk
2. Run in 2GB VRAM
3. Use only known methods
4. Be verifiable and deterministic
5. Act on a computer natively
6. Improve itself continuously

This is not science fiction. This is engineering. Make it work.
