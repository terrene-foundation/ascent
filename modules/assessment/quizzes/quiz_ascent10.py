# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""ASCENT Module 10 — AI-Resilient Assessment Questions

Alignment, RL & Governance
Covers: LoRA, QLoRA, SFT, DPO, GRPO, PPO, SAC, model merging,
        PACT GovernanceEngine, PactGovernedAgent, operating envelopes,
        D/T/R, AlignmentPipeline, AdapterRegistry, RLTrainer
"""

QUIZ = {
    "module": "ASCENT10",
    "title": "Alignment, RL & Governance",
    "questions": [
        # ── Section A: LoRA / SFT ───────────────────────────────────────
        {
            "id": "10.A.1",
            "lesson": "10.A",
            "type": "context_apply",
            "difficulty": "advanced",
            "question": (
                "Exercise 1 fine-tunes with LoRA (lora_r=16, target_modules=['q_proj', 'v_proj']). "
                "For a 7B model where q_proj has shape 4096×4096, calculate the total trainable "
                "parameters for q_proj and v_proj combined, and the reduction ratio vs full fine-tuning."
            ),
            "options": [
                "A) 131,072 trainable params per module × 2 modules = 262,144 per layer. Across 32 layers: 262K × 32 = 8.4M trainable. Full fine-tuning of q_proj+v_proj: 2 × 16.7M × 32 = 1.07B. Reduction for these layers: 128×. As a fraction of the full 7B model: 8.4M / 7B = 0.12% of total parameters are trainable.",
                "B) 65,536 trainable params total — LoRA only adds one matrix, not two",
                "C) 8,388,608 trainable params — LoRA trains 50% of the original parameters",
                "D) 16,384 trainable params — lora_r=16 means 16 parameters per module",
            ],
            "answer": "A",
            "explanation": (
                "Per target module: A matrix (4096 × 16) = 65,536 params + B matrix (16 × 4096) = 65,536. "
                "Total per module: 131,072. Two modules (q_proj + v_proj): 262,144 per layer. "
                "Across 32 transformer layers: 262,144 × 32 = 8,388,608 trainable parameters. "
                "Full fine-tuning: 2 × 4096 × 4096 × 32 = 1.07B parameters for q_proj + v_proj. "
                "Reduction: 1.07B / 8.39M ≈ 128×. "
                "Total model: 7B parameters, only 8.4M trainable = 0.12%. "
                "Optimizer memory: only 8.4M params need Adam states (m, v), saving ~95% GPU memory."
            ),
            "learning_outcome": "Calculate LoRA parameter reduction for a specific model architecture",
        },
        {
            "id": "10.A.2",
            "lesson": "10.A",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "Exercise 1's AlignmentConfig crashes with OOM on a 16GB GPU. "
                "The config has batch_size=16 and max_seq_length=2048. What change fixes it?"
            ),
            "code": (
                "config = AlignmentConfig(\n"
                "    method='sft',\n"
                "    base_model=os.environ['SFT_BASE_MODEL'],\n"
                "    lora_r=16,\n"
                "    lora_alpha=32,\n"
                "    batch_size=16,          # Too large\n"
                "    max_seq_length=2048,\n"
                "    gradient_checkpointing=False,  # Not enabled\n"
                ")\n"
            ),
            "options": [
                "A) Reduce lora_r from 16 to 4 — this is the main memory consumer",
                "B) Reduce batch_size to 2-4 and enable gradient_checkpointing=True. Activation memory scales as batch_size × seq_length × hidden_dim. At batch=16 and seq=2048, activations alone can consume 12+ GB. Gradient checkpointing recomputes activations during backward pass instead of storing them, trading ~30% compute for ~50% memory savings.",
                "C) Reduce max_seq_length to 128 — shorter sequences always fit in memory",
                "D) Switch from SFT to DPO — DPO uses less memory",
            ],
            "answer": "B",
            "explanation": (
                "Memory breakdown for SFT on a 7B model with batch=16, seq=2048: "
                "Model weights (fp16): 14 GB. LoRA weights: ~16 MB (negligible). "
                "Activations: batch × seq × layers × hidden × 2 bytes ≈ 16 × 2048 × 32 × 4096 × 2 ≈ 8.6 GB. "
                "Optimizer states: ~32 MB (only LoRA params). Total: ~23 GB → OOM on 16GB. "
                "Fix 1: batch_size=4 → activations ≈ 2.1 GB → total ~16.5 GB (tight fit). "
                "Fix 2: gradient_checkpointing=True → activations ≈ 1 GB → total ~15 GB (fits). "
                "Both together: comfortably fits with headroom for variable-length sequences."
            ),
            "learning_outcome": "Debug OOM errors in AlignmentConfig by adjusting batch size and gradient checkpointing",
        },
        {
            "id": "10.A.3",
            "lesson": "10.A",
            "type": "output_interpret",
            "difficulty": "advanced",
            "question": (
                "Exercise 1 produces this memory comparison table for fine-tuning a 7B model:\n\n"
                "| Method        | Model Weights | Trainable Params | Optimizer States | Total  |\n"
                "| Full FT       | 14 GB         | 14 GB            | 28 GB            | 56 GB  |\n"
                "| LoRA (r=16)   | 14 GB         | 8.4M (16.8 MB)   | 33.6 MB          | ~14 GB |\n"
                "| QLoRA (r=16)  | 3.5 GB        | 8.4M (16.8 MB)   | 33.6 MB          | ~3.5 GB|\n\n"
                "Why does QLoRA use only 3.5 GB for model weights instead of 14 GB, "
                "and why are optimizer states so dramatically different between Full FT and LoRA?"
            ),
            "options": [
                "A) QLoRA deletes 75% of the model weights — it only keeps the most important parameters",
                "B) QLoRA quantizes the frozen base model to 4-bit (14 GB fp16 ÷ 4 = 3.5 GB). LoRA adapters remain in fp16 for training precision. Optimizer states differ because Adam stores 2 states (m, v) per trainable parameter: Full FT has 7B trainable × 2 × 2 bytes = 28 GB. LoRA has 8.4M trainable × 2 × 2 bytes = 33.6 MB — an ~833× reduction. The base model weights require zero optimizer states because they are frozen.",
                "C) QLoRA uses model pruning to remove unnecessary layers, reducing from 14 GB to 3.5 GB",
                "D) The table is wrong — QLoRA cannot reduce memory below the LoRA baseline",
            ],
            "answer": "B",
            "explanation": (
                "Memory breakdown by component: "
                "(1) Model weights: Full FT and LoRA both store the base model in fp16 (7B × 2 bytes = 14 GB). "
                "QLoRA quantizes to NF4 (4-bit): 7B × 0.5 bytes = 3.5 GB. The model still works because "
                "4-bit NormalFloat preserves 99.7% of fp16 information for inference. "
                "(2) Trainable params: Full FT trains all 7B. LoRA/QLoRA train only low-rank adapters (8.4M). "
                "(3) Optimizer states: Adam maintains first moment (m) and second moment (v) for each "
                "trainable parameter. Full FT: 7B × 2 states × 2 bytes = 28 GB. "
                "LoRA/QLoRA: 8.4M × 2 states × 2 bytes = 33.6 MB — an ~833× reduction. "
                "QLoRA's breakthrough: 4-bit base model + fp16 adapters = fine-tuning a 7B model on a "
                "single 8 GB GPU (3.5 GB weights + 16.8 MB adapters + 33.6 MB optimizer ≈ 3.6 GB + activations)."
            ),
            "learning_outcome": "Interpret QLoRA memory savings from quantization of frozen weights and reduced optimizer states",
        },
        # ── Section B: DPO ──────────────────────────────────────────────
        {
            "id": "10.B.1",
            "lesson": "10.B",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 2 aligns with DPO using beta=0.1. You have 8,000 preference pairs "
                "and want to align a credit explanation model to be less technical. "
                "Should you use DPO or SFT + RLHF (PPO)?"
            ),
            "options": [
                "A) RLHF always produces better alignment than DPO",
                "B) DPO. With 8,000 preference pairs, DPO is more practical: it directly optimizes the preference objective without needing a separate reward model or PPO training loop. RLHF requires: (1) train a reward model, (2) run PPO against it — two unstable training stages. DPO collapses both into a single stable objective: L_DPO = -log σ(β(log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))).",
                "C) Neither — 8,000 pairs is too few for any alignment method",
                "D) SFT on the chosen responses only — preference pairs are unnecessary",
            ],
            "answer": "B",
            "explanation": (
                "DPO eliminates the reward model training step by showing that the optimal "
                "RLHF policy can be directly parameterized from preferences. "
                "Practical advantages: (1) One training run vs two (reward model + PPO). "
                "(2) No reward model architecture decisions. (3) More stable — PPO is notoriously "
                "sensitive to hyperparameters (clip range, KL coefficient, reward scaling). "
                "(4) 8,000 pairs is sufficient for DPO but marginal for training a good reward model. "
                "The beta parameter controls divergence from the reference policy: "
                "beta=0.1 allows moderate deviation. Higher beta → closer to reference."
            ),
            "learning_outcome": "Choose DPO over RLHF for stable preference alignment with limited data",
        },
        {
            "id": "10.B.2",
            "lesson": "10.B",
            "type": "output_interpret",
            "difficulty": "advanced",
            "question": (
                "Exercise 2 evaluates DPO-aligned vs base model on safety prompts. "
                "The DPO model scores 4.2/5 on helpfulness (base: 4.5) but 4.8/5 on safety "
                "(base: 2.1). Is this trade-off acceptable?"
            ),
            "options": [
                "A) No — any decrease in helpfulness means the alignment failed",
                "B) Yes — this is the expected alignment tax. DPO teaches the model to prefer safe responses over maximally helpful ones. A 0.3-point helpfulness decrease for a 2.7-point safety increase is an excellent trade-off. The alignment tax (small helpfulness reduction for large safety gain) is inherent to all alignment methods and is the entire point of alignment.",
                "C) The safety score improvement is suspicious — DPO cannot improve safety, only helpfulness",
                "D) Re-train with higher beta to eliminate the helpfulness decrease entirely",
            ],
            "answer": "B",
            "explanation": (
                "The alignment tax is well-documented: aligned models sacrifice some capability "
                "for improved safety/harmlessness. A 7% helpfulness decrease (4.5→4.2) for "
                "a 129% safety increase (2.1→4.8) is an excellent ratio. "
                "This happens because the preference data teaches: 'When safety and helpfulness "
                "conflict, prefer safety.' The model learns to refuse or caveat harmful requests "
                "instead of being maximally helpful regardless of consequences. "
                "Higher beta would actually REDUCE the alignment effect (keeping model closer to "
                "the unaligned reference), not eliminate the tax."
            ),
            "learning_outcome": "Evaluate alignment tax trade-off between helpfulness and safety",
        },
        {
            "id": "10.B.3",
            "lesson": "10.B",
            "type": "code_debug",
            "difficulty": "advanced",
            "question": (
                "Exercise 2's DPO training completes without errors, but the aligned model "
                "behaves identically to the base model. Evaluation shows 0% preference shift "
                "on the test set. Training loss decreased normally. What is wrong?"
            ),
            "code": (
                "config = AlignmentConfig(\n"
                "    method='dpo',\n"
                "    base_model=os.environ['DPO_BASE_MODEL'],\n"
                "    beta=50.0,        # Bug: beta far too high\n"
                "    lora_r=16,\n"
                "    lora_alpha=32,\n"
                "    learning_rate=5e-7,\n"
                "    num_epochs=3,\n"
                ")\n"
            ),
            "options": [
                "A) The learning rate is too low — increase to 1e-3 for faster convergence",
                "B) beta=50.0 is far too high. In DPO's loss function, beta controls the strength of the KL divergence constraint from the reference policy. High beta means the model is heavily penalized for deviating from the reference (unaligned) policy — effectively preventing any alignment. At beta=50, even a small policy change produces enormous KL penalty, so the optimizer learns to stay near the reference. Fix: beta=0.1-0.5 is the standard range. beta=0.1 allows moderate deviation; beta=0.5 for conservative alignment.",
                "C) DPO requires at least 10 epochs — 3 is not enough for any alignment",
                "D) The lora_r=16 is too small — the adapter cannot express the alignment changes",
            ],
            "answer": "B",
            "explanation": (
                "DPO loss = -log σ(β × (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x))). "
                "The beta parameter scales the log-ratio difference. At beta=50, the gradient signal "
                "is dominated by the KL constraint: any deviation from π_ref is penalized 50× more "
                "than the preference signal. The optimizer minimizes total loss by keeping π_θ ≈ π_ref "
                "(zero deviation = zero KL penalty), ignoring the preference data entirely. "
                "Training loss decreases because the model learns to perfectly predict the reference "
                "policy's outputs — but this is NOT alignment, it is reference copying. "
                "Standard beta values: 0.1 (aggressive alignment), 0.2-0.3 (balanced), 0.5 (conservative). "
                "Beta=50 is 100-500× too high."
            ),
            "learning_outcome": "Diagnose DPO alignment failure from excessive beta constraining policy deviation",
        },
        # ── Section C: RL ───────────────────────────────────────────────
        {
            "id": "10.C.1",
            "lesson": "10.C",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "Exercise 3 implements Q-learning on a grid world. The agent converges to "
                "a suboptimal policy: it takes the safe but long path (15 steps, no penalties) "
                "instead of the short path (5 steps through two cells with -1.0 penalty each). "
                "The goal gives +1.0. The discount factor is gamma=0.5. What is wrong?"
            ),
            "options": [
                "A) The learning rate is too high — the Q-values are oscillating",
                "B) gamma=0.5 is too low, making the distant goal nearly worthless. Short path value: -1.0 + 0.5×0 + 0.5²×0 + 0.5³×(-1.0) + 0.5⁴×1.0 = -1.0 - 0.125 + 0.0625 = -1.0625 (negative!). Long path: 0.5¹⁵ × 1.0 ≈ 0.00003 (tiny but non-negative). The agent rationally avoids the short path because penalties are immediate but the goal is heavily discounted. Fix: gamma=0.99 gives short path value ≈ -1.0 + 0.99³×(-1.0) + 0.99⁴×1.0 = -1.0 - 0.97 + 0.96 = -1.01 — still negative. With gamma=0.99 and a +10 goal: +6.1 (clearly positive).",
                "C) The epsilon-greedy exploration rate is too low — the agent never discovers the short path",
                "D) Q-learning cannot handle negative rewards — switch to SARSA",
            ],
            "answer": "B",
            "explanation": (
                "Discount factor gamma controls how much the agent values future rewards. "
                "At gamma=0.5, each step discounts by half. The short path traverses two -1.0 "
                "penalty cells: V_short = -1.0 + 0.5³×(-1.0) + 0.5⁴×1.0 = -1.0 - 0.125 + 0.0625 "
                "= -1.0625 (net negative). The long safe path: V_long = 0.5¹⁵ × 1.0 ≈ 0 (tiny but "
                "non-negative). The agent rationally prefers the long path because gamma=0.5 makes "
                "the goal too distant to offset the immediate penalties. "
                "Fix: increase gamma to 0.99 AND increase the goal reward so the discounted goal "
                "outweighs the penalties. The general principle: low gamma → myopic agents that "
                "avoid any short-term pain regardless of long-term gain."
            ),
            "learning_outcome": "Tune discount factor gamma for appropriate reward time horizon",
        },
        {
            "id": "10.C.2",
            "lesson": "10.C",
            "type": "context_apply",
            "difficulty": "advanced",
            "question": (
                "Exercise 4 trains PPO on inventory management. The reward curve shows: "
                "rapid improvement for 50 episodes, then a plateau at reward=850, while "
                "the heuristic baseline achieves 800. A colleague says 'only 6% better "
                "than a simple rule — RL is not worth it.' How do you respond?"
            ),
            "options": [
                "A) They're right — 6% is not worth the training complexity",
                "B) The 6% average improvement understates RL's value. Check the variance: PPO likely has lower reward variance (more consistent decisions). Also check edge cases: PPO may dramatically outperform the heuristic during demand spikes or supply disruptions where static rules fail. The heuristic uses a fixed threshold; PPO adapts its policy based on the current state.",
                "C) Train for more episodes — PPO always converges to the optimal solution eventually",
                "D) The reward function is wrong — redesign it to get a larger improvement",
            ],
            "answer": "B",
            "explanation": (
                "Average reward comparison misses two key metrics: "
                "(1) Variance: PPO with consistent decisions (std=20) vs heuristic with "
                "high variance (std=100) is operationally much better — fewer stockouts. "
                "(2) Tail performance: PPO adapts to unusual demand patterns (holiday spikes, "
                "supply chain disruptions) where the fixed heuristic threshold fails badly. "
                "The heuristic may average 800 but hit -500 during a supply crisis, "
                "while PPO adjusts ordering dynamically. "
                "In production, reliability matters more than average performance."
            ),
            "learning_outcome": "Evaluate RL policy beyond average reward using variance and tail performance",
        },
        {
            "id": "10.C.3",
            "lesson": "10.C",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 4 uses PPO for inventory management with discrete actions "
                "(order 0, 10, 50, or 100 units). A new requirement adds continuous "
                "actions: the agent must decide EXACTLY how many units to order (0-200, "
                "any integer). PPO with a softmax output over 201 discrete actions is "
                "impractical. What algorithm should you use instead?"
            ),
            "options": [
                "A) Keep PPO but bin the actions into 10 ranges — discretization always works",
                "B) Switch to SAC (Soft Actor-Critic). SAC is designed for continuous action spaces: the actor outputs a mean and standard deviation for a Gaussian policy, directly sampling continuous values. PPO CAN handle continuous actions but SAC is more sample-efficient in continuous spaces because its entropy regularization encourages exploration automatically. RLTrainer supports both: RLTrainer(algorithm='sac', action_space='continuous').",
                "C) Use Q-learning with a continuous Q-function — just replace the Q-table with a neural network",
                "D) Continuous actions are impossible in RL — always discretize the action space",
            ],
            "answer": "B",
            "explanation": (
                "Continuous action spaces require policies that output probability distributions "
                "over real-valued actions, not discrete softmax probabilities. "
                "SAC (Soft Actor-Critic): outputs μ and σ for a Gaussian → samples action ~ N(μ, σ²). "
                "Entropy bonus (H(π)) encourages exploration without epsilon-greedy heuristics. "
                "PPO can also handle continuous actions with a Gaussian policy head, but SAC's "
                "maximum entropy framework provides more stable training and better exploration "
                "in continuous spaces. "
                "For inventory management: SAC outputs 'order 73.2 units' directly, instead of "
                "choosing from 201 discrete bins. This matters when the optimal order quantity is "
                "sensitive to exact values (e.g., 73 units vs 70 units has meaningfully different cost). "
                "Kailash's RLTrainer(algorithm='sac') handles the continuous policy architecture."
            ),
            "learning_outcome": "Choose SAC over PPO for continuous action space RL problems",
        },
        # ── Section D: Model merging ────────────────────────────────────
        {
            "id": "10.D.1",
            "lesson": "10.D",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 5 merges SFT and DPO adapters. Linear merge gives F1=0.78, "
                "SLERP gives F1=0.82. Why does SLERP outperform linear interpolation?"
            ),
            "options": [
                "A) SLERP uses more computation during merging, producing a better result",
                "B) Linear interpolation W = α×W_sft + (1-α)×W_dpo can shrink weight magnitudes. At α=0.5, ||W|| ≤ 0.5×||W_sft|| + 0.5×||W_dpo||. This 'magnitude collapse' weakens the model. SLERP interpolates along the unit hypersphere, preserving ||W|| throughout the interpolation. The merged weights maintain their original scale, preserving model capacity.",
                "C) SLERP automatically selects the better model for each layer — it's not truly merging",
                "D) Linear merge requires the adapters to be trained on the same data; SLERP does not",
            ],
            "answer": "B",
            "explanation": (
                "Consider two unit vectors at 90°: linear interpolation at t=0.5 gives a vector "
                "with magnitude cos(45°) = 0.707 — a 30% magnitude reduction. "
                "SLERP maintains unit magnitude throughout the interpolation. "
                "For neural network weights, magnitude carries information: a shrunken weight "
                "matrix produces weaker activations, potentially losing learned patterns. "
                "SLERP preserves the 'energy' of both adapters while smoothly blending their "
                "directional information. Exercise 5 demonstrates this: linear merge's F1=0.78 "
                "reflects the capacity loss, while SLERP's F1=0.82 preserves full model strength."
            ),
            "learning_outcome": "Choose SLERP over linear merge to preserve weight magnitude during adapter merging",
        },
        {
            "id": "10.D.2",
            "lesson": "10.D",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "Exercise 5 registers merged adapters in AdapterRegistry. A student tries to "
                "merge three adapters (SFT + DPO + domain) but gets poor results. The SFT and "
                "DPO adapters were trained on the same base model, but the domain adapter was "
                "trained on a different base model. What went wrong?"
            ),
            "options": [
                "A) Three-way merging is not supported — only two adapters can be merged",
                "B) Adapters from different base models occupy different weight spaces and cannot be meaningfully merged. SFT and DPO adapters are corrections (ΔW) relative to the SAME base model W₀. Merging ΔW_sft + ΔW_dpo makes sense because both modify the same W₀. A domain adapter from a different W₀' produces ΔW_domain that is meaningless when applied to W₀.",
                "C) The merge weights were not set correctly — use 0.33 for each adapter",
                "D) AdapterRegistry does not support three adapters — only register two at a time",
            ],
            "answer": "B",
            "explanation": (
                "LoRA adapters are residual corrections: W_final = W_base + ΔW. "
                "When merging ΔW_1 and ΔW_2, we assume both correct the SAME W_base. "
                "If ΔW_domain was trained against a different W_base', then: "
                "W_base + ΔW_sft + ΔW_dpo + ΔW_domain ≠ anything meaningful, because "
                "ΔW_domain 'expects' W_base' underneath it. "
                "Solution: retrain the domain adapter on the same base model, or merge "
                "SFT+DPO first, then fine-tune the merged result on domain data."
            ),
            "learning_outcome": "Ensure adapter base model compatibility before merging in AdapterRegistry",
        },
        {
            "id": "10.D.3",
            "lesson": "10.D",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "Exercise 5 exports the merged model to ONNX and compares three variants:\n\n"
                "| Variant          | Size   | F1    | Latency |\n"
                "| fp16 (original)  | 14 GB  | 0.82  | 45ms    |\n"
                "| INT8 quantized   | 3.5 GB | 0.81  | 18ms    |\n"
                "| INT4 quantized   | 1.8 GB | 0.76  | 12ms    |\n\n"
                "Your production environment has 4 GB GPU memory and requires F1 ≥ 0.80. "
                "Which variant do you deploy?"
            ),
            "options": [
                "A) fp16 — always deploy the highest quality model",
                "B) INT8 quantized. fp16 at 14 GB does not fit in 4 GB GPU memory — eliminated. INT4 at F1=0.76 fails the ≥0.80 requirement — eliminated. INT8 at 3.5 GB fits the GPU, achieves F1=0.81 (above threshold), and runs at 18ms (2.5× faster than fp16). The 1-point F1 drop (0.82→0.81) is negligible for a 4× size reduction and 2.5× speed improvement.",
                "C) INT4 — smallest size and fastest latency are always the priority",
                "D) None — quantization always destroys model quality; request a larger GPU",
            ],
            "answer": "B",
            "explanation": (
                "Production deployment requires matching constraints to available options: "
                "(1) Memory constraint: 4 GB GPU eliminates fp16 (14 GB). "
                "(2) Quality constraint: F1 ≥ 0.80 eliminates INT4 (F1=0.76). "
                "(3) INT8 satisfies both: 3.5 GB < 4 GB, F1=0.81 ≥ 0.80. "
                "The INT8 quantization trade-off: each weight stored as 8-bit integer instead of "
                "16-bit float. With GPTQ group quantization (group_size=128), the effective "
                "compression is better than naive 8-bit: each group shares scale/zero-point "
                "parameters, yielding ~3.5 GB actual size. F1 drops only 0.01 because "
                "group quantization preserves most of the model's representational capacity. "
                "The 2.5× latency improvement (45ms → 18ms) comes from: smaller memory bandwidth, "
                "INT8 tensor core operations, and reduced cache pressure."
            ),
            "learning_outcome": "Select ONNX quantization level based on memory, quality, and latency constraints",
        },
        # ── Section E: Governance — PACT intro ─────────────────────────
        {
            "id": "10.E.1",
            "lesson": "10.E",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "Exercise 6 defines a PACT organization with D/T/R delegations. The "
                "model_trainer agent (clearance=confidential) tries to access restricted-level "
                "audit logs. GovernanceEngine blocks the request. The student asks: "
                "'But the trainer needs to see audit logs to debug training issues.' "
                "What is the governance-correct solution?"
            ),
            "options": [
                "A) Elevate model_trainer's clearance to restricted",
                "B) Create a new delegation. The chief_risk_officer (who has restricted clearance authority) delegates a 'training_audit' task to model_trainer with a SCOPED envelope: read-only access to training-related audit entries only, not all audit logs. This maintains least-privilege while enabling the legitimate use case. The D/T/R chain: CRO → training_audit → model_trainer.",
                "C) Disable the clearance check for the audit logs endpoint",
                "D) Have the risk_assessor agent read the logs and send a summary to model_trainer",
            ],
            "answer": "B",
            "explanation": (
                "PACT's D/T/R grammar solves this elegantly: "
                "1. The need is legitimate — trainer needs training audit data. "
                "2. Elevating clearance (A) violates least-privilege — trainer gets ALL restricted data. "
                "3. Disabling checks (C) removes governance entirely. "
                "4. A new delegation from CRO creates a scoped access path: "
                "   D=chief_risk_officer, T=training_audit, R=model_trainer "
                "   Envelope: read_only=True, filter='training_*', max_rows=1000 "
                "5. This is auditable: the audit trail shows WHO authorized WHAT access for WHOM. "
                "Option D (agent proxy) is a workaround, not governance."
            ),
            "learning_outcome": "Use scoped D/T/R delegations to grant least-privilege access",
        },
        {
            "id": "10.E.2",
            "lesson": "10.E",
            "type": "code_debug",
            "difficulty": "advanced",
            "question": (
                "Exercise 7 wraps agents with PactGovernedAgent. The governed agent has "
                "max_budget_usd=2.0 but processes 5 queries at $0.50 each ($2.50 total) "
                "without being blocked. The audit trail shows all 5 as 'allowed'. "
                "What is wrong with the governance configuration?"
            ),
            "code": (
                "governed = PactGovernedAgent(\n"
                "    agent=base_agent,\n"
                "    governance_engine=engine,\n"
                "    role='analyst',\n"
                "    max_budget_usd=2.0,\n"
                "    allowed_tools=['read_data', 'analyze_text'],\n"
                "    # Bug: missing clearance_level\n"
                ")\n"
            ),
            "options": [
                "A) max_budget_usd is per-query, not cumulative — set to 0.40 for a $2.00 total limit",
                "B) The budget enforcement checks each query independently against max_budget_usd, not cumulative spend. To enforce cumulative limits, the GovernanceEngine must be configured with budget tracking enabled. Additionally, missing clearance_level defaults to the highest level, bypassing data access controls. Always specify clearance_level explicitly.",
                "C) PactGovernedAgent budget enforcement only works with Delegate, not ReActAgent",
                "D) The governance_engine was not compiled — call engine.compile_org() first",
            ],
            "answer": "B",
            "explanation": (
                "Two issues: (1) Budget tracking must be explicitly enabled in GovernanceEngine "
                "for cumulative enforcement. Without it, each request is checked independently "
                "against max_budget_usd. 5 × $0.50 < $2.00 per-request, so all pass. "
                "With tracking: running total $0.50, $1.00, $1.50, $2.00, $2.50 → 5th blocked. "
                "(2) Missing clearance_level is a security gap — the default may be permissive. "
                "Always specify clearance_level to enforce data access boundaries. "
                "The audit trail showing 'allowed' for all 5 is the diagnostic clue — "
                "cumulative enforcement would show the 5th as 'blocked'."
            ),
            "learning_outcome": "Configure PactGovernedAgent with cumulative budget tracking and explicit clearance",
        },
        {
            "id": "10.E.3",
            "lesson": "10.E",
            "type": "output_interpret",
            "difficulty": "advanced",
            "question": (
                "Exercise 6's GovernanceEngine audit trail shows this sequence:\n\n"
                "```\n"
                "14:01:03 ALLOWED analyst → read_data(customers) [clearance=confidential]\n"
                "14:01:05 ALLOWED analyst → analyze_text(report_q3) [clearance=confidential]\n"
                "14:01:08 BLOCKED analyst → write_data(customers, {salary: ...}) [requires=restricted]\n"
                "14:01:08 ALLOWED analyst → read_data(products) [clearance=confidential]\n"
                "14:01:10 BLOCKED analyst → export_data(customers) [requires=restricted]\n"
                "14:01:10 ALLOWED analyst → analyze_text(report_q4) [clearance=confidential]\n"
                "```\n\n"
                "What pattern do you observe and what does it tell you about the agent's behavior?"
            ),
            "options": [
                "A) The analyst is behaving normally — blocks are expected for restricted operations",
                "B) The analyst is probing access boundaries. After being blocked on write_data(customers), it immediately tries export_data(customers) — a different operation on the same restricted resource. This retry-with-different-verb pattern suggests the agent is trying to find an allowed path to modify or extract customer data. The interleaved read/analyze calls between blocked attempts may be the agent 'disguising' its intent. GovernanceEngine correctly blocks both, but this pattern should trigger an escalation alert.",
                "C) The blocks are false positives — the analyst's clearance should include restricted data",
                "D) The audit trail is normal — two blocks out of six requests is a healthy ratio",
            ],
            "answer": "B",
            "explanation": (
                "Governance audit trails reveal behavioral patterns beyond individual allow/block decisions. "
                "The sequence shows: (1) analyst reads customer data (allowed — read is within clearance). "
                "(2) analyst tries to WRITE customer salary data (blocked — requires restricted clearance). "
                "(3) analyst reads a different table (potentially innocuous, or covering tracks). "
                "(4) analyst tries to EXPORT customer data (blocked — different verb, same restricted target). "
                "This probe-and-retry pattern with verb variation (write → export) is a known escalation "
                "signal in agent governance. The GovernanceEngine should: (1) Block the operations (done). "
                "(2) Flag the pattern for human review. (3) Consider reducing the agent's clearance "
                "if the pattern persists. PACT's verification gradient escalates from automated "
                "to human review based on exactly this kind of behavioral signal."
            ),
            "learning_outcome": "Interpret GovernanceEngine audit trails for agent boundary-probing behavior",
        },
        {
            "id": "10.E.4",
            "lesson": "10.E",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "Exercise 7's PactGovernedAgent can read public data but cannot read "
                "confidential data — even though its clearance_level is set to 'confidential'. "
                "The GovernanceEngine logs show: 'clearance check: agent_level=public, "
                "required=confidential → BLOCKED'. What is wrong?"
            ),
            "code": (
                "governed = PactGovernedAgent(\n"
                "    agent=base_agent,\n"
                "    governance_engine=engine,\n"
                "    role='analyst',\n"
                "    clearance_level='confidential',  # Set here...\n"
                "    max_budget_usd=5.0,\n"
                "    allowed_tools=['read_data', 'analyze_text'],\n"
                ")\n"
                "\n"
                "# But the org definition says:\n"
                "engine = GovernanceEngine(\n"
                "    org=Organization(\n"
                "        roles={\n"
                "            'analyst': Role(\n"
                "                clearance='public',  # ...overridden here\n"
                "                delegations=['read', 'analyze'],\n"
                "            )\n"
                "        }\n"
                "    )\n"
                ")\n"
            ),
            "options": [
                "A) PactGovernedAgent should be created before GovernanceEngine — the order matters",
                "B) The GovernanceEngine's Organization definition takes precedence over PactGovernedAgent's clearance_level parameter. The org defines role='analyst' with clearance='public'. When PactGovernedAgent sets clearance_level='confidential', the engine resolves the EFFECTIVE clearance by checking the org role definition — which says 'public'. Fix: update the org role definition to clearance='confidential', OR create a separate role with the correct clearance.",
                "C) 'confidential' is not a valid clearance level — use 'secret' instead",
                "D) The allowed_tools list overrides clearance_level — remove it to restore clearance",
            ],
            "answer": "B",
            "explanation": (
                "PACT governance follows a principle of organizational authority: the Organization "
                "definition is the source of truth for what each role can do. PactGovernedAgent's "
                "clearance_level is a REQUEST, not an override. The GovernanceEngine resolves: "
                "effective_clearance = min(agent_request, org_role_maximum). "
                "If the org says analyst has 'public' clearance, no agent can self-escalate to "
                "'confidential' — that would bypass the governance model entirely. "
                "Fix: the CRO or admin must update the org definition: "
                "Role('analyst', clearance='confidential'). "
                "This ensures clearance changes go through proper authorization channels, "
                "maintaining the governance audit trail. The diagnostic clue was the log showing "
                "agent_level=public despite the code saying 'confidential' — the engine resolved it."
            ),
            "learning_outcome": "Understand Organization role definitions override PactGovernedAgent clearance parameters",
        },
        # ── Section F: GovernanceEngine — org definition & access control ─
        {
            "id": "10.F.1",
            "lesson": "10.F",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "Exercise 6 compiles the SG FinTech AI Division YAML with GovernanceEngine. "
                "The compilation raises: 'CircularDelegationError: risk_assessor → data_analyst → risk_assessor'. "
                "Which section of the YAML is the root cause and what is the fix?"
            ),
            "code": (
                "delegations:\n"
                "  - delegator: 'chief_risk_officer'\n"
                "    task: 'risk_assessment'\n"
                "    responsible: 'risk_assessor'\n"
                "    envelope:\n"
                "      allowed_tools: ['read_data', 'audit_model', 'generate_report']\n"
                "\n"
                "  - delegator: 'risk_assessor'   # Bug: a Responsible cannot also be a Delegator\n"
                "    task: 'data_prep'\n"
                "    responsible: 'data_analyst'\n"
                "    envelope:\n"
                "      allowed_tools: ['read_data']\n"
                "\n"
                "  - delegator: 'data_analyst'    # Bug: closes the cycle back to risk_assessor\n"
                "    task: 'report_input'\n"
                "    responsible: 'risk_assessor'\n"
                "    envelope:\n"
                "      allowed_tools: ['generate_report']\n"
            ),
            "options": [
                "A) Rename the delegations to break the naming cycle — the names are the problem",
                "B) Remove the delegation from data_analyst → risk_assessor. In PACT's D/T/R grammar, an agent that is the Responsible in one chain can be a Delegator into a sub-chain (fan-out), but a Responsible CANNOT delegate back to an agent that already appears above it in the same chain. risk_assessor → data_analyst → risk_assessor creates a cycle that GovernanceEngine rejects during compile_org() because responsibility chains must form a DAG (directed acyclic graph) to guarantee traceable accountability.",
                "C) Set max_budget_usd=0 on one of the delegations to break the cycle at runtime",
                "D) Circular delegations are allowed — change GovernanceEngine to strict=False to suppress the error",
            ],
            "answer": "B",
            "explanation": (
                "GovernanceEngine.compile_org() validates the delegation graph as a DAG. "
                "A cycle (risk_assessor → data_analyst → risk_assessor) breaks the accountability "
                "invariant: if a task fails, it is impossible to trace accountability back to a unique "
                "human Delegator — the chain loops forever. "
                "PACT's rule: Responsible agents may fan out (delegate to sub-agents), but the "
                "resulting sub-chains must converge back to a human Delegator at the top, not to "
                "another agent already in the chain. "
                "Fix: remove the data_analyst → risk_assessor delegation entirely. If risk_assessor "
                "needs input from data_analyst, that dependency should be expressed within a single "
                "bounded task envelope, not as a separate D/T/R chain."
            ),
            "learning_outcome": "Identify circular delegation chains that violate PACT's DAG accountability requirement",
        },
        {
            "id": "10.F.2",
            "lesson": "10.F",
            "type": "output_interpret",
            "difficulty": "advanced",
            "question": (
                "Exercise 6's GovernanceEngine compiles the SG FinTech organization. "
                "The printed summary shows:\n\n"
                "  Agents registered: 6\n"
                "  Delegations: 4\n"
                "  Departments: 3\n"
                "  Clearance levels: public < internal < confidential < restricted\n"
                "  Compilation validates: clearance levels are monotonically decreasing down chains\n\n"
                "A student then creates a delegation from risk_assessor (clearance=restricted) "
                "to a new sub-agent temp_helper (clearance=confidential) with the envelope "
                "allowed_data_clearance='restricted'. The compilation succeeds. "
                "Is this governance configuration correct?"
            ),
            "options": [
                "A) Yes — the compilation passed, so it is correct",
                "B) No. The envelope grants temp_helper access to restricted data, but temp_helper's clearance is only confidential. GovernanceEngine's compile_org() validates that clearances decrease monotonically down the chain (restricted → confidential is correct), but the envelope's allowed_data_clearance must also not exceed the Responsible agent's own clearance. allowed_data_clearance='restricted' exceeds temp_helper's 'confidential' clearance — this envelope should be rejected or auto-clamped to 'confidential'.",
                "C) No — a Responsible agent can never have a lower clearance than its Delegator under any circumstances",
                "D) Yes — the envelope overrides the agent's clearance level; this is how privileged delegation works in PACT",
            ],
            "answer": "B",
            "explanation": (
                "PACT operates on the principle that envelopes cannot grant access beyond what the "
                "Responsible agent is cleared to handle. The compilation validates chain monotonicity "
                "(restricted → confidential is a valid decrease), but a correct implementation also "
                "validates that allowed_data_clearance ≤ Responsible's clearance. "
                "temp_helper at confidential clearance receiving a restricted-data envelope is a "
                "privilege escalation: the agent gains access to data it was never vetted to handle. "
                "The intended pattern: either elevate temp_helper's clearance to restricted (with "
                "proper authorization), or reduce the envelope to allowed_data_clearance='confidential'. "
                "The fact that compilation succeeds here points to a missing envelope validation "
                "step — students should verify envelope bounds manually when working near clearance "
                "boundaries."
            ),
            "learning_outcome": "Verify envelope allowed_data_clearance does not exceed Responsible agent's clearance level",
        },
        {
            "id": "10.F.3",
            "lesson": "10.F",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 6 maps the SG FinTech AI Division to Singapore's AI Verify and "
                "the EU AI Act. Your organization is a Singapore company with EU customers. "
                "Your AI system makes automated loan approval decisions. "
                "Which two PACT governance features are mandatory and why?"
            ),
            "options": [
                "A) Budget limits and tool restrictions — these prevent runaway costs",
                "B) D/T/R delegation chains (for accountability tracing) and immutable audit trails (for record-keeping). EU AI Act Art. 13 requires high-risk AI systems to provide transparency on decision logic. Art. 14 requires human oversight mechanisms. Singapore AI Verify requires accountability — every AI decision must trace to a human who authorized it. D/T/R satisfies both: the chain traces loan_decision_agent back to the credit_director (human Delegator). The immutable audit trail satisfies Art. 12 record-keeping and MAS TRM 7.5 for financial services.",
                "C) Clearance levels and operating envelopes — these restrict what data the AI can see",
                "D) GovernanceEngine compilation and budget cascade — these prevent policy violations",
            ],
            "answer": "B",
            "explanation": (
                "Loan approval AI is high-risk under EU AI Act Annex III. Mandatory requirements: "
                "(1) Art. 9 Risk Management: operating envelopes define what the AI can do — addresses this. "
                "(2) Art. 12 Record-keeping: PACT's immutable audit trail logs every decision with reason codes. "
                "(3) Art. 13 Transparency: D/T/R chains make the authorization path human-readable. "
                "(4) Art. 14 Human oversight: the Delegator (credit_director) is a named human accountable "
                "for the agent's decisions — satisfying the 'human in the loop' requirement. "
                "Singapore AI Verify's accountability principle maps directly to D/T/R: "
                "every action by an AI agent must be attributable to a responsible human decision-maker. "
                "Budget and clearance controls are important but are not the primary regulatory requirements "
                "for high-risk automated financial decisions."
            ),
            "learning_outcome": "Map PACT D/T/R and audit trail to EU AI Act and AI Verify regulatory requirements",
        },
        # ── Section G: PactGovernedAgent — runtime enforcement ─────────
        {
            "id": "10.G.1",
            "lesson": "10.G",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "Exercise 7 wraps a ReActAgent with PactGovernedAgent. "
                "The governed_analyst is allowed to call read_data and analyze_text. "
                "When asked to 'Read the sg_company_reports dataset', it calls read_data successfully. "
                "When asked to 'Summarize findings', it calls analyze_text successfully. "
                "But when the ReActAgent internally chains both calls in a single reasoning step, "
                "the second call (analyze_text) raises GovernanceViolation. What is wrong?"
            ),
            "code": (
                "governed_analyst = PactGovernedAgent(\n"
                "    agent=base_agent,\n"
                "    governance_engine=engine,\n"
                "    role='analyst',\n"
                "    max_budget_usd=2.0,\n"
                "    allowed_tools=['read_data', 'analyze_text'],\n"
                "    clearance_level='internal',\n"
                "    per_tool_budget_usd=0.50,  # Bug: per-tool limit applied cumulatively\n"
                ")\n"
            ),
            "options": [
                "A) ReActAgent cannot be wrapped with PactGovernedAgent — use Delegate instead",
                "B) per_tool_budget_usd=0.50 applies a $0.50 limit per individual tool call. The read_data call costs $0.45 (within limit). But in a chained reasoning step, the ReActAgent calls read_data and analyze_text sequentially without returning control — the governance layer sees a single 'tool chain' cost of $0.45+$0.40=$0.85, which exceeds per_tool_budget_usd=0.50. Fix: either remove per_tool_budget_usd and rely on max_budget_usd cumulative tracking, or raise per_tool_budget_usd to accommodate chained calls.",
                "C) The clearance_level='internal' blocks analyze_text — it requires confidential clearance",
                "D) analyze_text is not in the ReActAgent's tool list — add it to the base_agent tools",
            ],
            "answer": "B",
            "explanation": (
                "PactGovernedAgent intercepts each tool call from the wrapped agent. "
                "When ReActAgent chains multiple tool calls in a single reasoning step, "
                "the governance layer sees them either as individual calls (each checked against "
                "per_tool_budget_usd) or as a grouped action depending on implementation. "
                "The diagnostic is that both tools work individually (single-call tests pass) but "
                "fail when chained — this points to a budget parameter that accumulates across the "
                "chain rather than resetting per call. "
                "Best practice: use max_budget_usd for cumulative session limits, and avoid "
                "per_tool_budget_usd unless you explicitly need per-call cost caps (e.g., to prevent "
                "a single expensive vector search from consuming the whole budget)."
            ),
            "learning_outcome": "Distinguish per_tool_budget_usd from max_budget_usd in PactGovernedAgent configuration",
        },
        {
            "id": "10.G.2",
            "lesson": "10.G",
            "type": "context_apply",
            "difficulty": "advanced",
            "question": (
                "Exercise 7 defines governed_analyst (allowed_tools=['read_data','analyze_text']) "
                "and governed_deployer (allowed_tools=['read_data','deploy_model']). "
                "Both share the same base_agent (ReActAgent). "
                "A new requirement: the analyst must be able to trigger deployment "
                "after completing analysis — but ONLY if the analysis shows risk < 5%. "
                "How should this be implemented using PACT?"
            ),
            "options": [
                "A) Add 'deploy_model' to governed_analyst's allowed_tools with a condition parameter",
                "B) Do not modify governed_analyst. Instead, create a new governed_reviewer PactGovernedAgent with a conditional delegation: the human Delegator (chief_ml_officer) pre-authorizes deployment only when risk_score < 0.05. The workflow calls governed_analyst for analysis, checks the risk_score from the audit trail output, and if below threshold, invokes governed_deployer. This maintains role separation: analyst assesses, deployer acts. The human Delegator's pre-authorization covers the conditional path. Adding deploy_model to analyst's envelope collapses the roles and creates an ungoverned conditional path.",
                "C) Modify the base_agent to check risk_score before calling deploy_model internally",
                "D) Use engine.check_access() with a conditional parameter before calling governed_deployer",
            ],
            "answer": "B",
            "explanation": (
                "PACT's role separation is not about convenience — it maps to regulatory accountability. "
                "If analyst and deployer are the same role, a single agent can both assess and act, "
                "removing the four-eyes check that high-risk AI systems require. "
                "The correct pattern: governed_analyst produces a structured output including risk_score. "
                "The orchestrator (not an agent) evaluates risk_score < 0.05 and decides to call "
                "governed_deployer — this decision happens at the human-authorized orchestration layer. "
                "The audit trail shows: analyst decision → orchestrator check → deployer action, "
                "with the human Delegator's pre-authorization covering all three steps. "
                "Option A (conditional envelope) is not a PACT primitive — it would require custom "
                "enforcement logic that bypasses the governance engine's standard checks."
            ),
            "learning_outcome": "Maintain role separation in PactGovernedAgent instead of adding conditional tool access",
        },
        {
            "id": "10.G.3",
            "lesson": "10.G",
            "type": "output_interpret",
            "difficulty": "advanced",
            "question": (
                "Exercise 7 extracts the audit trail after running governed_analyst and "
                "governed_deployer. The audit shows:\n\n"
                "  Analyst audit entries: 6\n"
                "  Deployer audit entries: 3\n"
                "  Total allowed: 7\n"
                "  Total blocked: 2\n\n"
                "The two blocked entries are:\n"
                "  [14:02:11] deploy_model: blocked — tool not in allowed_tools\n"
                "  [14:02:14] delete_data: blocked — tool not in allowed_tools\n\n"
                "Both blocked entries appear in the ANALYST's audit trail. "
                "The student says 'PactGovernedAgent is broken — analyst shouldn't even be "
                "attempting these calls.' Is the student correct?"
            ),
            "options": [
                "A) Yes — a correctly configured governed agent should never attempt disallowed tools",
                "B) No. The ReActAgent's language model generates tool calls based on its reasoning about the task, independent of governance restrictions. The LLM inside the agent may decide 'to complete this analysis I should deploy the model' and generate a deploy_model call. PactGovernedAgent intercepts ALL tool calls from the wrapped agent — including ones the LLM generates that exceed the envelope. The blocks are correct behavior, not a bug. The audit trail is working as designed: governance is fail-closed (attempts are blocked and logged), and the blocked attempts are audited, providing a full behavioral record.",
                "C) Yes — update the system prompt to tell the analyst agent it cannot deploy",
                "D) No — the blocks are expected, but they indicate the base_agent's tool list is too permissive and deploy_model should be removed from it",
            ],
            "answer": "B",
            "explanation": (
                "This is a fundamental architectural distinction: the LLM inside the agent reasons "
                "about what tools to call based on the task description and its training. It does not "
                "know (and should not know) about governance restrictions — that would require "
                "modifying the LLM's behavior via system prompt, which is fragile and gameable. "
                "PactGovernedAgent is the enforcement layer: it sits between the LLM's tool call "
                "decisions and actual tool execution, intercepting and blocking disallowed calls. "
                "This separation is intentional: the LLM focuses on reasoning, governance focuses "
                "on enforcement. Blocked calls appearing in the audit trail are the correct outcome — "
                "they show the governance layer caught an overreach. "
                "Option C (system prompt restriction) is a soft control; Option B (governance intercept) "
                "is a hard control. PACT uses hard controls because soft controls can be bypassed "
                "through prompt injection or emergent reasoning."
            ),
            "learning_outcome": "Understand PactGovernedAgent as a hard enforcement layer independent of the LLM's reasoning",
        },
        # ── Section H: Capstone — full-stack governed ML system ─────────
        {
            "id": "10.H.1",
            "lesson": "10.H",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 8 integrates kailash-align (AdapterRegistry), kailash-pact "
                "(GovernanceEngine, PactGovernedAgent), kailash-kaizen (BaseAgent, Signature), "
                "and kailash-nexus (Nexus) into one governed ML system. "
                "A student suggests: 'We could skip PactGovernedAgent and just add governance "
                "checks inside the QAAgent's forward method.' What is wrong with this approach?"
            ),
            "options": [
                "A) It would work — inlining governance is simpler and equivalent",
                "B) Inline governance inside the agent violates the separation of concerns that makes the system auditable and upgradeable. Three specific problems: (1) Auditability: PACT's GovernanceEngine writes to an immutable audit chain. Inline checks are application logs — not immutable, not structured, and not exportable for regulatory compliance. (2) Consistency: governance checks inside the agent run only when that specific agent is called. PactGovernedAgent wraps ANY BaseAgent uniformly — the same enforcement layer applies across qa_agent, admin_agent, and any future agents. (3) Upgradability: regulatory requirements change. Updating one GovernanceEngine config updates all governed agents. Inlined checks require hunting through agent code for each rule.",
                "C) BaseAgent.forward() is not the right hook — use BaseAgent.run() instead",
                "D) The performance overhead of PactGovernedAgent is too high for production — inline is faster",
            ],
            "answer": "B",
            "explanation": (
                "The capstone exercise exists specifically to show how the frameworks compose: "
                "kailash-align handles model lifecycle, kailash-kaizen handles agent reasoning, "
                "kailash-pact handles governance enforcement, kailash-nexus handles deployment. "
                "Each layer has a defined responsibility. Inlining governance into the agent "
                "conflates two responsibilities: 'how do I reason about this task' (agent) and "
                "'am I authorized to act on my reasoning' (governance). "
                "The regulatory implications are concrete: EU AI Act Art. 12 requires records "
                "in a form that regulators can inspect. Application logs are not sufficient — "
                "they can be deleted, modified, and are not structured for compliance queries. "
                "PACT's audit chain is immutable by design, exportable in compliance report formats, "
                "and covers all governed agents uniformly. "
                "The performance cost of PactGovernedAgent (one access check per tool call) is "
                "negligible compared to the LLM inference cost it wraps."
            ),
            "learning_outcome": "Justify PactGovernedAgent as a separate governance layer rather than inline checks for auditability and regulatory compliance",
        },
        {
            "id": "10.H.2",
            "lesson": "10.H",
            "type": "code_debug",
            "difficulty": "advanced",
            "question": (
                "Exercise 8 deploys the governed QA system via Nexus. The handler works "
                "correctly when called directly (await handle_qa(...)), but when called through "
                "Nexus's API channel, the governance audit trail is empty after each request. "
                "What is the bug?"
            ),
            "code": (
                "app = Nexus()\n"
                "app.register(handle_qa)\n"
                "app.start()\n"
                "\n"
                "async def handle_qa(question: str, role: str = 'qa') -> dict:\n"
                "    agent = governed_qa if role == 'qa' else governed_admin\n"
                "    result = await agent.run(question=question)\n"
                "    return {'answer': result.answer, 'governed': True}\n"
                "\n"
                "# Bug: new PactGovernedAgent instance created per Nexus worker\n"
                "# governed_qa = PactGovernedAgent(...) defined inside a factory function\n"
                "# called once per request — audit state is lost between requests\n"
            ),
            "options": [
                "A) Nexus does not support async handlers — convert handle_qa to synchronous",
                "B) PactGovernedAgent instances are being created fresh for each Nexus request (defined inside a factory). Each new instance has an empty audit trail. The governed_qa and governed_admin must be module-level singletons shared across all Nexus requests — exactly as shown in the exercise solution where they are defined at module level before app.register(). The GovernanceEngine singleton accumulates the audit trail across all requests; per-request instances lose all history when garbage collected.",
                "C) Call app.create_session() before handle_qa to initialize audit state",
                "D) The audit trail is only populated after app.stop() is called — it flushes on shutdown",
            ],
            "answer": "B",
            "explanation": (
                "This is a classic stateful-object lifecycle bug. PactGovernedAgent accumulates "
                "the audit trail in its instance state. If the instance is created per-request "
                "(inside a factory or inside handle_qa itself), the audit trail exists only for "
                "the duration of that request and is garbage collected afterward. "
                "The exercise solution intentionally defines governed_qa and governed_admin at "
                "module level (outside any function) so they are singletons: one instance "
                "shared across all requests, with a single audit trail accumulating over the "
                "system's lifetime. "
                "The diagnostic: audit trail empty after each request (not after a few) means "
                "state is never accumulating — pointing to per-request instantiation rather than "
                "a flush/export problem. "
                "The GovernanceEngine must also be a singleton for the same reason: its compiled "
                "org graph and access control state must persist across requests."
            ),
            "learning_outcome": "Ensure PactGovernedAgent and GovernanceEngine are module-level singletons for persistent audit state across Nexus requests",
        },
        {
            "id": "10.H.3",
            "lesson": "10.H",
            "type": "context_apply",
            "difficulty": "advanced",
            "question": (
                "Exercise 8's compliance report maps the governed ML system to four regulatory "
                "frameworks: EU AI Act, Singapore AI Verify, MAS TRM, and PACT's own D/T/R "
                "accountability grammar. A new requirement arrives: the system must now produce "
                "a real-time compliance dashboard showing live governance status. "
                "Which two kailash frameworks do you combine, and what does each contribute?"
            ),
            "options": [
                "A) kailash-ml (DriftMonitor) + kailash-align (AdapterRegistry) — monitor model drift and track adapter versions",
                "B) kailash-pact (GovernanceEngine) + kailash-nexus (Nexus). GovernanceEngine exposes get_audit_trail() and check_access() calls that yield live governance state: current budget spend per agent, blocked action counts, clearance violation attempts. Nexus registers a /governance/status handler that polls GovernanceEngine in real time and serves it as a REST endpoint (API channel) or MCP resource (MCP channel). This requires no new frameworks — the capstone already imports both. The dashboard handler calls governed_qa.get_audit_trail() and governed_admin.get_audit_trail() and aggregates the results.",
                "C) kailash-kaizen (Delegate) + kailash-dataflow (DataFlow) — agent queries a database of audit events",
                "D) kailash-align (AlignmentPipeline) + kailash-nexus (Nexus) — expose alignment metrics via API",
            ],
            "answer": "B",
            "explanation": (
                "The capstone already has both components wired: GovernanceEngine with compiled org "
                "and PactGovernedAgent instances accumulating audit state, plus Nexus for multi-channel "
                "deployment. The dashboard requires: (1) data source (GovernanceEngine.get_audit_trail(), "
                "PactGovernedAgent.get_budget_status()) and (2) delivery channel (Nexus handler). "
                "Implementation: add a /governance/status handler to the Nexus app that aggregates "
                "qa_audit = governed_qa.get_audit_trail() and admin_audit = governed_admin.get_audit_trail(), "
                "computes summary statistics (allowed/blocked counts, budget consumed, clearance violations), "
                "and returns a structured dict. Nexus serves this via API (/governance/status GET), "
                "CLI (> governance status), and MCP (resource: governance/status). "
                "No additional frameworks needed — this is the composability payoff of the capstone design."
            ),
            "learning_outcome": "Compose GovernanceEngine audit data with Nexus multi-channel deployment for real-time compliance dashboards",
        },
    ],
}
