# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""ASCENT Module 7 — AI-Resilient Assessment Questions

Deep Learning: Architecture-Driven Feature Engineering
Covers: neural networks, hidden layers, activations, loss functions,
        backpropagation, optimizers, CNNs, OnnxBridge, InferenceServer
"""

QUIZ = {
    "module": "ASCENT07",
    "title": "Deep Learning",
    "questions": [
        # ── Lesson 7.1: Linear regression as NN ─────────────────────────
        {
            "id": "7.1.1",
            "lesson": "7.1",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student implements gradient descent for linear regression as a "
                "single neuron. Training loss decreases for 3 epochs then explodes "
                "to infinity. The learning rate is 0.01. What is wrong?"
            ),
            "code": (
                "# Forward pass\n"
                "y_pred = w * x + b\n"
                "loss = sum((y_pred - y) ** 2) / n  # MSE\n"
                "\n"
                "# Backward pass\n"
                "dw = sum(2 * (y_pred - y) * x) / n\n"
                "db = sum(2 * (y_pred - y)) / n\n"
                "\n"
                "# Update (BUG)\n"
                "w = w + lr * dw\n"
                "b = b + lr * db\n"
            ),
            "options": [
                "A) MSE loss should divide by 2n, not n — the factor of 2 causes the gradient to be too large",
                "B) The update rule adds the gradient instead of subtracting it. Gradient descent MINIMIZES loss: w = w - lr * dw. Adding moves uphill.",
                "C) The learning rate 0.01 is too high for any gradient descent — reduce to 1e-6",
                "D) dw and db should use absolute value |y_pred - y| instead of (y_pred - y)",
            ],
            "answer": "B",
            "explanation": (
                "Gradient descent follows the NEGATIVE gradient direction to minimize loss. "
                "The update rule must be w = w - lr * dw (subtract). "
                "Adding the gradient moves parameters in the direction of INCREASING loss, "
                "causing divergence. The loss may decrease briefly due to stochastic noise "
                "before exploding. This is one of the most common DL bugs."
            ),
            "learning_outcome": "Implement correct gradient descent update rule for a single neuron",
        },
        {
            "id": "7.1.2",
            "lesson": "7.1",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "In Exercise 1, you implement y = wx + b as a neural network and compare "
                "gradient descent with the OLS closed-form solution. Your GD converges to "
                "w=0.482, b=3.21 while OLS gives w=0.485, b=3.19. The GD loss is 0.0312 "
                "while OLS loss is 0.0310. Should you worry about the difference?"
            ),
            "options": [
                "A) Yes — the difference indicates a bug in the backpropagation. GD should converge to the exact same solution as OLS.",
                "B) Yes — GD is always worse than OLS for linear regression and should never be used.",
                "C) No — GD converged close to the optimum. Small differences arise from learning rate, number of epochs, and floating-point precision. For linear regression, OLS is exact; GD is iterative and approximate. The 0.6% difference in w is negligible.",
                "D) No — the losses are identical (0.03), so the parameters must be equivalent.",
            ],
            "answer": "C",
            "explanation": (
                "GD is iterative — it approaches the optimum asymptotically. With finite epochs "
                "and a fixed learning rate, it may not reach the exact minimum. The 0.6% parameter "
                "difference and 0.6% loss difference confirm convergence to a near-optimal solution. "
                "For linear regression specifically, OLS computes the exact minimum analytically. "
                "GD's value is that it scales to non-linear models where no closed-form exists. "
                "The exercise demonstrates that DL starts from this familiar foundation."
            ),
            "learning_outcome": "Understand GD convergence properties vs analytical solutions",
        },
        # ── Lesson 7.2: Hidden layers ───────────────────────────────────
        {
            "id": "7.2.1",
            "lesson": "7.2",
            "type": "architecture_decision",
            "difficulty": "intermediate",
            "question": (
                "Your dataset has two features (x1, x2) and a binary label where the "
                "positive class forms a ring around the negative class (like a donut). "
                "A single-layer network achieves 50% accuracy. What change is needed?"
            ),
            "options": [
                "A) Increase the learning rate — the model needs more aggressive updates to find the decision boundary",
                "B) Add a hidden layer with non-linear activation. The ring decision boundary is not linearly separable — a hidden layer can learn a non-linear transformation that makes it separable (similar to XOR requiring a hidden layer).",
                "C) Add more features by computing x1² and x2² manually — neural networks cannot learn polynomial features",
                "D) Use MSE loss instead of cross-entropy — MSE works better for circular decision boundaries",
            ],
            "answer": "B",
            "explanation": (
                "A ring/donut pattern is not linearly separable — no single hyperplane can separate it. "
                "This is the same fundamental problem as XOR. A hidden layer with non-linear activation "
                "(e.g., ReLU) learns a feature transformation where the classes become separable. "
                "The universal approximation theorem guarantees a sufficiently wide single hidden layer "
                "can approximate any continuous function, but in practice, deeper networks learn such "
                "transformations more efficiently."
            ),
            "learning_outcome": "Identify when hidden layers are necessary for non-linear decision boundaries",
        },
        {
            "id": "7.2.2",
            "lesson": "7.2",
            "type": "output_interpret",
            "difficulty": "advanced",
            "question": (
                "You visualize the decision boundary of a 2-hidden-layer network trained on "
                "the spiral dataset from Exercise 2. The boundary is very jagged with sharp "
                "corners that perfectly trace every training point. Test accuracy is 62% while "
                "training accuracy is 99%. What happened and what would you change?"
            ),
            "options": [
                "A) The model is underfitting — add more hidden layers to increase capacity",
                "B) The learning rate was too low — the model needs to explore more of the loss landscape",
                "C) The model is overfitting — the jagged boundary memorizes training noise. Add dropout (Exercise 7), reduce hidden layer width, or add L2 regularization to smooth the boundary.",
                "D) The spiral dataset is too hard — switch to a simpler dataset",
            ],
            "answer": "C",
            "explanation": (
                "99% train / 62% test is classic overfitting. The jagged boundary means the model "
                "memorized individual training points instead of learning the underlying spiral pattern. "
                "Solutions: (1) Dropout randomly zeros neurons during training, forcing redundancy. "
                "(2) Reducing width limits capacity to memorize. (3) L2 regularization penalizes large "
                "weights, encouraging smoother boundaries. The gap between train/test accuracy (37 points) "
                "is the overfitting signal."
            ),
            "learning_outcome": "Diagnose overfitting from train/test accuracy gap and boundary visualization",
        },
        # ── Lesson 7.3: DL power hierarchy ────────────────────────────
        {
            "id": "7.3.1",
            "lesson": "7.3",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "You need to classify satellite images into 50 land-use categories. "
                "Each image is 256×256×3. A colleague proposes a wide single-hidden-layer "
                "network (1 layer, 8192 neurons). You propose a deep network (6 layers, "
                "512 neurons each). Both have ~100M parameters. Which is better and why?"
            ),
            "options": [
                "A) The wide network — universal approximation theorem guarantees a single hidden layer can represent any function, so depth adds no value",
                "B) The deep network. While a single wide layer CAN represent any function (UAT), it may need exponentially more neurons to do so. Depth enables hierarchical feature composition: layer 1 learns edges, layer 2 learns textures, layer 3 learns parts, layers 4-6 learn objects and scenes. This compositional hierarchy matches the structure of visual data — each layer reuses lower-level features, requiring far fewer total parameters for the same representational power.",
                "C) Both are equivalent — total parameter count is all that matters for network capacity",
                "D) The wide network — deep networks always suffer from vanishing gradients and cannot be trained",
            ],
            "answer": "B",
            "explanation": (
                "The depth vs width trade-off is central to the DL power hierarchy (Lesson 7.3). "
                "A single wide layer can theoretically represent any function (UAT), but may need "
                "exponentially many neurons. A deep network composes simple functions hierarchically: "
                "f(x) = f_6(f_5(f_4(f_3(f_2(f_1(x)))))). Each layer transforms representations, "
                "building from edges → textures → parts → objects. For 256×256 images with 50 "
                "categories, this compositionality is critical — a flat network must independently "
                "learn every pixel-to-category mapping, while a deep network reuses intermediate "
                "features. Modern techniques (ReLU, BatchNorm, skip connections) solve the vanishing "
                "gradient concern, making depth practical."
            ),
            "learning_outcome": "Choose depth over width for hierarchical feature learning in vision tasks",
        },
        # ── Lesson 7.4: Activations ────────────────────────────────────
        {
            "id": "7.4.1",
            "lesson": "7.4",
            "type": "code_debug",
            "difficulty": "advanced",
            "question": (
                "A student trains a 5-layer network with sigmoid activations. After 100 "
                "epochs, layers 1-2 have gradient magnitudes of ~1e-8 while layers 4-5 "
                "have gradients of ~0.01. What is this problem called and what fix does "
                "Exercise 3 demonstrate?"
            ),
            "options": [
                "A) Exploding gradients — add gradient clipping to cap gradients at 1.0",
                "B) Dead neurons — the sigmoid output is stuck at 0; switch to Leaky ReLU",
                "C) Vanishing gradients — sigmoid squashes to (0,1) and its max derivative is 0.25. Across 5 layers: 0.25⁵ ≈ 0.001. Early layers barely update. Fix: replace sigmoid with ReLU (derivative = 1 for positive inputs) and use He initialization.",
                "D) Learning rate is too small — increase from 0.01 to 0.1 to compensate for small gradients",
            ],
            "answer": "C",
            "explanation": (
                "Vanishing gradients occur when activation derivatives are consistently < 1. "
                "Sigmoid's max derivative is 0.25 (at x=0), so gradients shrink by at least 4× "
                "per layer. After 5 layers: gradient ≈ 0.25⁵ = ~0.001× the output gradient. "
                "ReLU has derivative exactly 1 for positive inputs, allowing gradients to flow "
                "unchanged through layers. He initialization (scale = sqrt(2/fan_in)) ensures "
                "activations maintain variance across layers. Exercise 3 demonstrates this "
                "by comparing gradient magnitudes per layer with sigmoid vs ReLU."
            ),
            "learning_outcome": "Diagnose vanishing gradients from per-layer gradient magnitudes",
        },
        {
            "id": "7.4.2",
            "lesson": "7.4",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "In Exercise 3, you compare sigmoid, ReLU, and GELU on the same architecture. "
                "ReLU converges in 30 epochs. GELU converges in 25 epochs with slightly better "
                "final accuracy. Sigmoid converges in 80 epochs. Why is GELU faster than ReLU "
                "despite both solving the vanishing gradient problem?"
            ),
            "options": [
                "A) GELU has a larger maximum derivative than ReLU, allowing bigger gradient updates",
                "B) GELU is smoother than ReLU around x=0. ReLU has a hard kink (derivative jumps from 0 to 1), while GELU's smooth curve x·Φ(x) provides better gradient signals near zero, enabling finer-grained updates in the critical transition region.",
                "C) GELU uses less memory than ReLU because it doesn't need to store the activation mask",
                "D) GELU converges faster only on this specific dataset — in general, ReLU is always faster",
            ],
            "answer": "B",
            "explanation": (
                "GELU (Gaussian Error Linear Unit) is defined as x·Φ(x) where Φ is the Gaussian CDF. "
                "Unlike ReLU's hard threshold at x=0, GELU smoothly transitions, providing non-zero "
                "gradients for slightly negative inputs. This smooth approximation of a stochastic "
                "regularizer helps the optimizer navigate the loss landscape more efficiently. "
                "GELU is the default activation in modern transformers (BERT, GPT) precisely because "
                "this smoothness improves convergence on a wide range of tasks."
            ),
            "learning_outcome": "Compare activation function convergence properties from Exercise 3 results",
        },
        # ── Lesson 7.5: Loss functions ──────────────────────────────────
        {
            "id": "7.5.1",
            "lesson": "7.5",
            "type": "architecture_decision",
            "difficulty": "intermediate",
            "question": (
                "You are building a 10-class image classifier (Fashion-MNIST). A colleague "
                "suggests using MSE loss because 'it worked fine for regression.' Exercise 4 "
                "shows MSE achieves 85% accuracy while CrossEntropy achieves 92% on the same "
                "architecture. Why does CrossEntropy outperform MSE for classification?"
            ),
            "options": [
                "A) MSE requires one-hot encoded labels which are wasteful; CrossEntropy uses integer labels directly",
                "B) CrossEntropy's gradient is (predicted - target), which provides strong gradients when the model is confident but wrong. MSE's gradient (2×(pred-target)×sigmoid_derivative) includes sigmoid_derivative which shrinks near 0 and 1, slowing learning when the model is most wrong.",
                "C) MSE can only be used for binary classification, not multi-class",
                "D) CrossEntropy uses the log function which makes the loss landscape convex for any model",
            ],
            "answer": "B",
            "explanation": (
                "The key difference is gradient behavior when the model is wrong. "
                "CrossEntropy with softmax gives gradient = (predicted - target), which is LARGE "
                "when the model confidently predicts the wrong class. "
                "MSE with sigmoid gives gradient = 2(pred-target) × σ'(z), where σ'(z) is near 0 "
                "when the output is near 0 or 1. So MSE's gradient VANISHES precisely when the "
                "model is most wrong and needs the biggest correction. "
                "Exercise 4 demonstrates this: MSE loss plateaus early while CE continues improving."
            ),
            "learning_outcome": "Choose appropriate loss function based on gradient dynamics for classification",
        },
        {
            "id": "7.5.2",
            "lesson": "7.5",
            "type": "code_debug",
            "difficulty": "advanced",
            "question": (
                "A student initializes a 10-layer ReLU network with all weights set to 0.01 "
                "(constant initialization). After 50 epochs, all neurons in each layer output "
                "identical values. What is this problem and what initialization from Exercise 4 "
                "fixes it?"
            ),
            "options": [
                "A) The learning rate is too small — increase to break symmetry",
                "B) Symmetry breaking failure. If all weights are identical, all neurons compute identical gradients and make identical updates forever. Use He initialization: w ~ N(0, sqrt(2/fan_in)) — random values break symmetry while the sqrt(2/fan_in) scale prevents vanishing/exploding activations in ReLU networks.",
                "C) Constant 0.01 is too small — use constant 1.0 instead to ensure activations are large enough",
                "D) The bias terms should be initialized to 1.0, not 0.0 — bias breaks the symmetry",
            ],
            "answer": "B",
            "explanation": (
                "Symmetry breaking is essential: if neurons in a layer start identical, they receive "
                "identical gradients and remain identical forever — effectively a single neuron "
                "replicated N times, wasting capacity. Random initialization breaks this symmetry. "
                "He initialization (scale = sqrt(2/fan_in)) is specifically designed for ReLU networks: "
                "the factor of 2 compensates for ReLU zeroing out negative inputs. "
                "Xavier initialization (sqrt(1/fan_in)) is optimal for sigmoid/tanh. "
                "Exercise 4 shows 10-layer networks diverge with constant init but converge with He."
            ),
            "learning_outcome": "Apply correct weight initialization strategy for deep ReLU networks",
        },
        # ── Lesson 7.6: Backpropagation ─────────────────────────────────
        {
            "id": "7.6.1",
            "lesson": "7.6",
            "type": "code_debug",
            "difficulty": "advanced",
            "question": (
                "In Exercise 5, you implement gradient checking to verify your backprop. "
                "For most parameters, the relative error between analytical and numerical "
                "gradients is < 1e-5. But for one weight, the relative error is 0.23. "
                "What does this tell you?"
            ),
            "options": [
                "A) The numerical gradient is inaccurate — decrease epsilon from 1e-5 to 1e-10",
                "B) Floating point precision limits mean errors up to 0.5 are acceptable",
                "C) Relative error of 0.23 (23%) indicates a bug in the analytical gradient for that weight. The chain rule was likely applied incorrectly for that parameter's computation path. Since other gradients pass, the error is localized — check the backward pass for that specific layer/connection.",
                "D) The model has converged and gradients near zero cause unstable relative errors",
            ],
            "answer": "C",
            "explanation": (
                "Gradient checking compares: analytical gradient (from backprop) vs numerical gradient "
                "((f(x+ε) - f(x-ε)) / 2ε). Relative error > 1e-2 indicates a bug. At 0.23 (23%), "
                "the analytical gradient is clearly wrong for that parameter. Common causes: "
                "(1) forgot to apply chain rule through activation derivative, "
                "(2) transposed weight matrix in gradient computation, "
                "(3) wrong variable used in gradient update. "
                "Since other parameters pass, the issue is localized to one computation path."
            ),
            "learning_outcome": "Use gradient checking to identify backpropagation implementation errors",
        },
        {
            "id": "7.6.2",
            "lesson": "7.6",
            "type": "context_apply",
            "difficulty": "advanced",
            "question": (
                "Exercise 5 demonstrates vanishing gradients in a 10-layer sigmoid network. "
                "You measure gradient norms per layer: [0.0001, 0.0003, 0.001, 0.004, 0.015, "
                "0.06, 0.2, 0.8, 3.1, 12.0] (layer 1 to 10). After switching to ReLU + He init, "
                "the norms become [0.8, 0.9, 1.1, 0.95, 1.0, 1.05, 0.98, 1.02, 1.1, 1.0]. "
                "What pattern do you observe in each case?"
            ),
            "options": [
                "A) Sigmoid: gradients grow toward the output; ReLU: gradients are random",
                "B) Sigmoid: gradients decay exponentially toward input layers (factor ~4× per layer = 0.25 from sigmoid derivative). Layer 1 gets gradient 120,000× smaller than layer 10. ReLU + He init: gradients maintain roughly constant magnitude (~1.0) across all layers, confirming healthy gradient flow.",
                "C) Both cases show healthy training — gradient magnitudes don't matter, only their direction",
                "D) Sigmoid gradients are too small; ReLU gradients are too large — both need gradient clipping",
            ],
            "answer": "B",
            "explanation": (
                "Sigmoid case: gradient ratio layer10/layer1 = 12.0/0.0001 = 120,000×. "
                "The exponential decay (~4× per layer due to sigmoid max derivative 0.25) means "
                "early layers learn at 1/120,000th the speed of later layers. "
                "ReLU + He case: max/min ratio = 1.1/0.8 = 1.375×. Near-constant gradients mean "
                "all layers learn at similar speeds. He initialization (sqrt(2/fan_in)) maintains "
                "activation variance = 1.0 across layers, which maintains gradient variance too. "
                "This is the fundamental insight Exercise 5 demonstrates."
            ),
            "learning_outcome": "Interpret per-layer gradient norms to diagnose training health",
        },
        # ── Lesson 7.8-7.9: Optimizers ──────────────────────────────────
        {
            "id": "7.9.1",
            "lesson": "7.9",
            "type": "output_interpret",
            "difficulty": "intermediate",
            "question": (
                "Exercise 6 compares SGD, SGD+Momentum, and Adam on MNIST. After 5 epochs: "
                "SGD loss=1.8, Momentum loss=0.9, Adam loss=0.4. The SGD loss curve oscillates "
                "wildly between epochs. What causes the oscillation and why does momentum fix it?"
            ),
            "options": [
                "A) SGD oscillates because the learning rate is too high; momentum slows down the effective learning rate",
                "B) SGD updates based on each mini-batch, causing noisy gradient estimates. In narrow ravines (common in loss landscapes), the gradient points across the ravine (oscillating) rather than along it. Momentum accumulates past gradients, canceling oscillations while reinforcing consistent direction — like a heavy ball rolling downhill.",
                "C) SGD uses the wrong loss function; Adam automatically selects the correct one",
                "D) The oscillation is normal and harmless — SGD will converge to the same loss given enough epochs",
            ],
            "answer": "B",
            "explanation": (
                "Mini-batch SGD computes gradients on a random subset, introducing noise. "
                "In narrow ravines of the loss landscape (common near saddle points), gradients "
                "oscillate perpendicular to the optimal path. Momentum (v = β×v + g; w -= lr×v) "
                "averages these oscillations: perpendicular components cancel, parallel components "
                "reinforce. The result is faster progress along the ravine. "
                "β=0.9 means 90% of the previous velocity carries forward, "
                "smoothing over ~1/(1-β)=10 past gradients."
            ),
            "learning_outcome": "Explain momentum's role in damping gradient oscillations",
        },
        {
            "id": "7.9.2",
            "lesson": "7.9",
            "type": "architecture_decision",
            "difficulty": "intermediate",
            "question": (
                "You are starting a new DL project with no prior hyperparameter knowledge. "
                "Exercise 6 shows Adam converges fastest with default hyperparameters. "
                "When should you prefer SGD+Momentum over Adam?"
            ),
            "options": [
                "A) Never — Adam is strictly better in all cases",
                "B) When you need reproducible results — SGD is deterministic while Adam is not",
                "C) When you have time for extensive hyperparameter tuning and want the best final performance. Research shows SGD+Momentum with carefully tuned learning rate + cosine schedule can generalize better than Adam on some tasks (image classification, language models). Adam converges faster but may find sharper minima.",
                "D) When your model has fewer than 1 million parameters — Adam's overhead is too high for small models",
            ],
            "answer": "C",
            "explanation": (
                "Adam is the best default optimizer: it adapts per-parameter learning rates and "
                "converges quickly with minimal tuning. However, multiple studies show SGD+Momentum "
                "with properly tuned learning rate + cosine annealing can achieve better generalization "
                "(flatter minima) on tasks like ImageNet classification and language model training. "
                "The trade-off: Adam = fast prototyping, minimal tuning. SGD+Momentum = potentially "
                "better final performance with more tuning effort. Start with Adam, switch to SGD "
                "if you need that last 0.5% accuracy."
            ),
            "learning_outcome": "Choose between Adam and SGD+Momentum based on project constraints",
        },
        # ── Lesson 7.11: CNNs ───────────────────────────────────────────
        {
            "id": "7.11.1",
            "lesson": "7.11",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "A student's CNN from Exercise 7 has conv(3×3, 4 filters, padding=0) → pool(2×2) "
                "on a 28×28 input. They set the fully-connected layer input size to 4×14×14=784. "
                "The model crashes with a dimension mismatch. What is the correct FC input size?"
            ),
            "options": [
                "A) 4 × 28 × 28 = 3136 — pooling doesn't change spatial dimensions",
                "B) 4 × 13 × 13 = 676. Without padding, conv(3×3) on 28×28 gives 26×26 (not 28×28). Pool(2×2) on 26×26 gives 13×13. With 4 filters: 4 × 13 × 13 = 676.",
                "C) 4 × 12 × 12 = 576 — conv reduces by 3 and pool by 2",
                "D) 1 × 14 × 14 = 196 — only one filter output is passed to the FC layer",
            ],
            "answer": "B",
            "explanation": (
                "Output size formula: out = (in - kernel + 2×padding) / stride + 1. "
                "Conv: (28 - 3 + 0) / 1 + 1 = 26. Pool: 26 / 2 = 13. "
                "With 4 filters: 4 × 13 × 13 = 676. "
                "The student assumed padding=1 (which would give 28→28→14), but the code has padding=0. "
                "This is the most common CNN dimension bug. "
                "Always trace spatial dimensions through each layer before writing the FC layer."
            ),
            "learning_outcome": "Calculate CNN output dimensions through conv and pooling layers",
        },
        {
            "id": "7.11.2",
            "lesson": "7.11",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 7 exports a CNN to ONNX via OnnxBridge and then serves it via "
                "InferenceServer in Exercise 8. Your model runs at 15ms/prediction in Python "
                "and 3ms/prediction via ONNX Runtime. A colleague asks: 'Why not just keep "
                "the Python model?' What is your answer?"
            ),
            "options": [
                "A) Python is fine for production — ONNX is only useful for non-Python environments",
                "B) ONNX Runtime applies graph-level optimizations (operator fusion, constant folding) and uses platform-native execution (CPU SIMD, GPU kernels) without Python interpreter overhead. The 5× speedup matters at scale: 1M daily predictions saves 3.3 hours of compute. Plus ONNX is portable — same model runs on mobile, browser, or server.",
                "C) ONNX reduces model size by 10× through automatic quantization during export",
                "D) Python models cannot be served over HTTP; only ONNX models work with InferenceServer",
            ],
            "answer": "B",
            "explanation": (
                "ONNX Runtime provides: (1) Graph optimizations — fuse adjacent operations like "
                "Conv+BatchNorm+ReLU into a single kernel. (2) Platform-native execution — "
                "vectorized CPU instructions or GPU kernels without Python GIL. (3) Portability — "
                "same .onnx file runs on any ONNX runtime (C++, JS, Java, mobile). "
                "The 5× speedup (15ms → 3ms) compounds: at 1M predictions/day, that's "
                "4.2 hours vs 0.8 hours of compute. OnnxBridge.export() handles the conversion; "
                "InferenceServer wraps it in an HTTP endpoint."
            ),
            "learning_outcome": "Justify ONNX export for production deployment via OnnxBridge",
        },
        # ── Lesson 7.10: Dropout ────────────────────────────────────────
        {
            "id": "7.10.1",
            "lesson": "7.10",
            "type": "output_interpret",
            "difficulty": "intermediate",
            "question": (
                "After training a network with dropout=0.5, a student runs the same input "
                "through the model 10 times and gets 10 DIFFERENT predictions. They then "
                "switch to model.eval() mode and get the same prediction every time, but "
                "all predictions are ~50% smaller in magnitude than the training outputs. "
                "What explains both observations?"
            ),
            "options": [
                "A) The model has a bug — predictions should be identical in both training and eval mode",
                "B) During training, dropout randomly zeros 50% of neurons per forward pass, so each run activates a different random subset — producing different outputs. In eval mode, dropout is disabled (all neurons active), but each neuron's output is scaled by (1-p)=0.5 to compensate for the fact that twice as many neurons are now active. If the implementation uses inverted dropout (scale by 1/p during training), eval outputs are correct without scaling. The student's model likely uses standard dropout without inverted scaling.",
                "C) The 10 different predictions indicate the model has not converged — train for more epochs",
                "D) model.eval() disables gradient computation, which changes the forward pass outputs",
            ],
            "answer": "B",
            "explanation": (
                "Dropout has fundamentally different behavior at train vs test time: "
                "Training: each neuron is randomly kept with probability p (or zeroed with probability "
                "1-p). This means each forward pass uses a different random subnetwork — hence 10 "
                "different predictions for the same input. "
                "Eval/Test: all neurons are active. Expected activation doubles (since during training "
                "only 50% were active). Two approaches: (1) Standard dropout: multiply outputs by p=0.5 "
                "at test time (what the student sees). (2) Inverted dropout: multiply by 1/p=2.0 during "
                "training so test time needs no adjustment. Most frameworks use inverted dropout by "
                "default. The student's 50%-magnitude outputs confirm standard dropout without test-time "
                "scaling."
            ),
            "learning_outcome": "Explain dropout behavior difference between training and evaluation modes",
        },
        # ── Lesson 7.3: Activation functions (additional) ───────────────
        {
            "id": "7.3.2",
            "lesson": "7.3",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "In Exercise 3, a student implements the GELU activation. Their version "
                "produces correct values for x=0 and x=2 but diverges for x=-3. "
                "Which line contains the bug?"
            ),
            "code": (
                "def gelu(z: float) -> float:\n"
                "    # Option A (student version)\n"
                "    return 0.5 * z * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (z + 0.044715 * z ** 2)))\n"
                "\n"
                "def gelu_correct(z: float) -> float:\n"
                "    # Option B (solution)\n"
                "    return 0.5 * z * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (z + 0.044715 * z ** 3)))\n"
            ),
            "options": [
                "A) The outer coefficient should be 0.25, not 0.5 — the Gaussian CDF integrates to 0.5 at x=0",
                "B) The exponent on z inside the tanh is z**2 instead of z**3. The correct GELU approximation uses the cubic term 0.044715·z³. For small positive z the error is minor, but for large negative z (x=-3), z² stays positive while z³ is negative, causing the tanh argument to point the wrong direction.",
                "C) math.sqrt(2.0 / math.pi) should be math.sqrt(math.pi / 2.0) — the constant is inverted",
                "D) The tanh should be replaced with sigmoid — GELU uses Φ(x) which is the sigmoid function",
            ],
            "answer": "B",
            "explanation": (
                "The GELU approximation formula from the original paper is: "
                "0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3))). "
                "The exponent must be 3, not 2. For z=-3: "
                "Correct: 0.044715 * (-3)^3 = 0.044715 * (-27) = -1.207 (large negative, tanh → -1, GELU → ~0). "
                "Bug: 0.044715 * (-3)^2 = 0.044715 * 9 = +0.402 (positive, tanh → +1 incorrectly). "
                "This is precisely why Exercise 3 tests the implementation at z=-3 — it reveals the z² vs z³ error "
                "that is invisible for small values near zero."
            ),
            "learning_outcome": "Identify the cubic term in the GELU approximation formula",
        },
        {
            "id": "7.3.3",
            "lesson": "7.3",
            "type": "output_interpret",
            "difficulty": "intermediate",
            "question": (
                "Exercise 3 prints gradient statistics for each activation function. "
                "You see this output:\n\n"
                "  sigmoid: mean|grad|=0.1562, zero_fraction=0.0%\n"
                "  relu:    mean|grad|=0.5021, zero_fraction=49.8%\n"
                "  gelu:    mean|grad|=0.5834, zero_fraction=0.0%\n\n"
                "ReLU has 49.8% zero gradients but still converges faster than sigmoid. "
                "How do you explain this apparent contradiction?"
            ),
            "options": [
                "A) The gradient statistics are wrong — ReLU cannot have 49.8% zeros if it converges",
                "B) 49.8% zero gradients for ReLU is expected: inputs drawn from N(0,1) have ~50% below zero. But the NON-ZERO gradients are all exactly 1.0 (no squashing), while sigmoid's non-zero gradients average 0.156 (max 0.25). The active neurons in ReLU get full gradient signal while sigmoid neurons get at most 25%. Mean gradient tells the wrong story — what matters is the magnitude of the non-zero gradients reaching early layers.",
                "C) GELU is always slower than ReLU because zero_fraction=0.0% means all neurons are active and compute more expensive updates",
                "D) The mean gradient is what determines convergence speed — sigmoid (0.156) is fastest since it has the lowest variance gradients",
            ],
            "answer": "B",
            "explanation": (
                "The key insight from Exercise 3's gradient analysis: for inputs drawn from N(0,1) "
                "(which is approximately what activations look like with He init), ~50% are below zero "
                "so ReLU zeros those gradients. But the other 50% get gradient exactly 1.0. "
                "Sigmoid's mean gradient of 0.156 means ALL gradients are reduced — the best case "
                "is 0.25 and the worst case approaches 0. Across 5+ layers, ReLU's 50%-active "
                "neurons still pass stronger gradient signal than sigmoid's 100%-active but squashed neurons. "
                "GELU's zero_fraction=0.0% and mean=0.583 confirms it provides the strongest average signal "
                "of the three, which explains its fastest convergence observed in Exercise 3."
            ),
            "learning_outcome": "Interpret gradient statistics to explain convergence differences between activations",
        },
        # ── Lesson 7.6: Optimizers and scheduling (additional) ──────────
        {
            "id": "7.6.3",
            "lesson": "7.6",
            "type": "code_debug",
            "difficulty": "advanced",
            "question": (
                "A student copies the Adam implementation from Exercise 6 but gets "
                "unexpectedly large updates in the first few steps. The bias correction "
                "is the suspected cause. What is wrong?"
            ),
            "code": (
                "# Student's Adam update (one parameter shown)\n"
                "t_step += 1\n"
                "m = beta1 * m + (1 - beta1) * grad\n"
                "v = beta2 * v + (1 - beta2) * grad ** 2\n"
                "# BUG: bias correction applied incorrectly\n"
                "m_hat = m * (1 - beta1 ** t_step)\n"
                "v_hat = v * (1 - beta2 ** t_step)\n"
                "param -= lr * m_hat / (math.sqrt(v_hat) + eps)\n"
            ),
            "options": [
                "A) beta1 and beta2 should be swapped — beta2 corrects the first moment, beta1 the second",
                "B) The bias correction DIVIDES by (1 - beta^t), not MULTIPLIES. At t=1, (1-0.9^1)=0.1, so dividing by 0.1 amplifies m by 10x to correct for the cold-start bias. The student multiplies by 0.1, shrinking the estimate by 10x instead — producing near-zero effective updates in early steps.",
                "C) The eps term should be inside the square root: math.sqrt(v_hat + eps)",
                "D) t_step should start at 0, not be incremented before the update — off-by-one causes divergence",
            ],
            "answer": "B",
            "explanation": (
                "Adam's bias correction formula is m_hat = m / (1 - beta1^t). "
                "At step t=1 with beta1=0.9: denominator = 1 - 0.9 = 0.1, so m_hat = m / 0.1 = 10×m. "
                "This amplification corrects for m being initialized at 0 — at t=1, m = (1-0.9)×grad = 0.1×grad, "
                "so m_hat = 0.1×grad / 0.1 = grad. The correction restores the true first-moment estimate. "
                "The student's version m_hat = m × (1-0.9^1) = m × 0.1 = 0.01×grad — a 100x underestimate "
                "that collapses all early updates to near zero. This matches the implementation in Exercise 6 "
                "where bc1 = 1 - beta1**t_step is used as the DIVISOR: m_hat = mW2[j][k] / bc1."
            ),
            "learning_outcome": "Implement Adam bias correction as division, not multiplication",
        },
        {
            "id": "7.6.4",
            "lesson": "7.6",
            "type": "context_apply",
            "difficulty": "intermediate",
            "question": (
                "Exercise 6 implements cosine annealing with linear warmup. The schedule "
                "prints: LR at step 0=0.000000, step 50=0.000500, step 100=0.001000 (peak), "
                "step 500=0.000501, step 999=0.000001. A student wants to double the peak LR "
                "to 0.002. They change only max_lr=0.002. What is the complete set of outputs "
                "they should now expect at those same four steps?"
            ),
            "options": [
                "A) step 0=0.000000, step 50=0.001000, step 100=0.002000, step 500=0.001001, step 999=0.000001",
                "B) step 0=0.002000, step 50=0.001000, step 100=0.001000, step 500=0.001001, step 999=0.000001",
                "C) step 0=0.000000, step 50=0.001000, step 100=0.002000, step 500=0.002000, step 999=0.002000",
                "D) step 0=0.000000, step 50=0.000500, step 100=0.002000, step 500=0.001001, step 999=0.000001",
            ],
            "answer": "A",
            "explanation": (
                "The cosine_schedule function from Exercise 6 has two phases: "
                "(1) Warmup [0, warmup_steps): LR = max_lr * step / warmup_steps. "
                "At step 50: 0.002 * 50/100 = 0.001000. This scales linearly with max_lr. "
                "(2) Cosine decay [warmup_steps, total_steps): min_lr + 0.5*(max_lr-min_lr)*(1+cos(π×progress)). "
                "At step 500 (progress=0.4444): 1e-6 + 0.5*(0.002-1e-6)*(1+cos(0.4444π)) = ~0.001001. "
                "At step 999 (progress≈1.0): cos(π)=-1, so LR → min_lr = 1e-6. "
                "Only option A correctly shows the warmup slope doubling (step 50: 0.001000) "
                "and the decay endpoint staying at min_lr=1e-6 regardless of max_lr."
            ),
            "learning_outcome": "Trace cosine schedule output values given a change to max_lr",
        },
        # ── Lesson 7.7: CNNs and OnnxBridge ─────────────────────────────
        {
            "id": "7.7.1",
            "lesson": "7.7",
            "type": "code_debug",
            "difficulty": "intermediate",
            "question": (
                "In Exercise 7, a student implements conv2d. When they apply the horizontal "
                "edge detector kernel [[-1,-1,-1],[0,0,0],[1,1,1]] to a 28×28 image with "
                "padding=1, they expect a 28×28 output but get 30×30. What is wrong in "
                "their padding implementation?"
            ),
            "code": (
                "def conv2d(image, kernel, stride=1, padding=0):\n"
                "    h, w = len(image), len(image[0])\n"
                "    kh, kw = len(kernel), len(kernel[0])\n"
                "    if padding > 0:\n"
                "        # BUG: padding applied wrong\n"
                "        padded = [[0.0] * (w + padding) for _ in range(h + padding)]\n"
                "        for i in range(h):\n"
                "            for j in range(w):\n"
                "                padded[i + padding][j + padding] = image[i][j]\n"
                "        image = padded\n"
                "        h, w = len(image), len(image[0])\n"
                "    out_h = (h - kh) // stride + 1\n"
                "    out_w = (w - kw) // stride + 1\n"
            ),
            "options": [
                "A) The output dimensions out_h and out_w should use ceiling division, not floor division",
                "B) The padded array adds 'padding' columns/rows but should add '2 * padding'. padding=1 means one zero-row on each of the four sides. The student's code creates a (29×29) padded array, places the original image starting at (1,1), leaving only a right/bottom border — no top/left border. The correct size is (h + 2*padding) × (w + 2*padding) = 30×30, then the output formula gives (30-3)//1+1=28. The student's (29-3)//1+1=27 output would also be wrong.",
                "C) stride=1 is wrong for a 28×28 image — stride should equal the kernel size (3)",
                "D) The image coordinates i+padding and j+padding are off by one — should be i+padding-1 and j+padding-1",
            ],
            "answer": "B",
            "explanation": (
                "Padding adds zeros around the image border. padding=1 means one row/column on EACH side: "
                "top, bottom, left, right. So the padded size is (h + 2×1) × (w + 2×1) = 30×30. "
                "The solution in Exercise 7 uses exactly this: "
                "padded = [[0.0] * (w + 2*padding) for _ in range(h + 2*padding)]. "
                "The student uses (h + padding) = 29, missing one full side. This causes the image to be "
                "placed with a top-left offset (1,1) but no bottom-right border, so the convolution "
                "center reaches outside the original image region and gives wrong spatial coverage. "
                "The '28→28 with padding=1' identity holds only with the correct 2×padding expansion."
            ),
            "learning_outcome": "Apply correct 2*padding formula in conv2d zero-padding implementation",
        },
        {
            "id": "7.7.2",
            "lesson": "7.7",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 7 exports a trained CNN using OnnxBridge.export() with "
                "input_shape=(1, 1, 28, 28). A student wants to export their model "
                "to serve predictions on a mobile app. They call bridge.validate() "
                "and get max_diff=1.2e-6. Should they proceed with deployment?"
            ),
            "options": [
                "A) No — any non-zero max_diff means the ONNX model is incorrect and must be re-exported",
                "B) Yes — max_diff=1.2e-6 is within acceptable floating-point tolerance. ONNX Runtime uses 32-bit float by default while the original model may use 64-bit. Differences of 1e-6 to 1e-5 are normal numerical precision artifacts, not errors. The bridge.validate() call in Exercise 7 is specifically designed to catch real discrepancies (>1e-3) that would indicate export failure.",
                "C) No — mobile deployment requires quantization first; the full-precision ONNX model is too large",
                "D) Yes — but only after converting max_diff from absolute to relative error and checking it is below 0.01%",
            ],
            "answer": "B",
            "explanation": (
                "OnnxBridge.validate() checks that the ONNX model produces outputs numerically equivalent "
                "to the original. Floating-point precision differences between frameworks and data types "
                "(float64 vs float32) routinely produce max_diff values of 1e-7 to 1e-5 — these are "
                "not errors. A max_diff of 1.2e-6 on a classification task (softmax outputs in [0,1]) "
                "means the probabilities differ in the 6th decimal place, which will NEVER change "
                "the argmax (predicted class). The Exercise 7 solution prints this metric precisely "
                "to teach students the threshold: differences < 1e-4 are safe, > 1e-2 indicate "
                "a real export problem. Mobile deployment is indeed the ONNX use case — portable, "
                "no Python runtime needed."
            ),
            "learning_outcome": "Interpret OnnxBridge.validate() max_diff output to approve ONNX export",
        },
        # ── Lesson 7.8: Capstone pipeline ────────────────────────────────
        {
            "id": "7.8.1",
            "lesson": "7.8",
            "type": "code_debug",
            "difficulty": "advanced",
            "question": (
                "A student runs the Exercise 8 capstone pipeline. The TrainingPipeline "
                "and ModelRegistry steps succeed, but InferenceServer crashes with "
                "AttributeError: 'InferenceServer' object has no attribute 'predict'. "
                "Their code is shown below. What is wrong?"
            ),
            "code": (
                "# Exercise 8 capstone — Task 5\n"
                "bridge = OnnxBridge()\n"
                "onnx_path = bridge.export(\n"
                "    model=result.model,\n"
                "    input_shape=(1, len(pixel_cols)),\n"
                "    output_path='fashion_mnist_cnn.onnx',\n"
                ")\n"
                "\n"
                "# BUG: student's deployment code\n"
                "server = InferenceServer(model_path=onnx_path, port=8090)\n"
                "prediction = server.predict(sample)  # crashes here\n"
            ),
            "options": [
                "A) model_path should be the pickle bytes, not the .onnx file path — InferenceServer takes a binary artifact",
                "B) server.predict() must be called inside an async function after awaiting server.start(). InferenceServer is async throughout: the server must be started before it can accept predictions. The solution in Exercise 8 uses 'async def deploy_and_test()' with 'await server.start()' before 'await server.predict(sample)'.",
                "C) The port 8090 is already in use — change to port 8091",
                "D) InferenceServer.predict() takes a DataFrame, not a list — wrap sample in pl.DataFrame first",
            ],
            "answer": "B",
            "explanation": (
                "InferenceServer is an async component. The Exercise 8 solution wraps all server interaction "
                "in 'async def deploy_and_test()' and runs it with asyncio.run(deploy_and_test()). "
                "Inside that function: 'await server.start()' brings the HTTP server online, then "
                "'prediction = await server.predict(sample)' sends a request to the running server. "
                "Calling server.predict() synchronously without starting the server first means there is "
                "no running server to receive the prediction request — hence the AttributeError (the "
                "async method is not callable as a regular method). "
                "This is the same async pattern used for OnnxBridge.export() and ModelRegistry.register_model() "
                "throughout the capstone — all kailash-ml I/O operations are async."
            ),
            "learning_outcome": "Apply async start() before predict() when deploying with InferenceServer",
        },
        {
            "id": "7.8.2",
            "lesson": "7.8",
            "type": "architecture_decision",
            "difficulty": "advanced",
            "question": (
                "Exercise 8 calls registry.promote_model(name='fashion_mnist_cnn', "
                "version=version.version, target_stage='production') immediately after "
                "registering. A student asks: 'Why register AND promote separately — why "
                "not just register with stage=production directly?' What is the correct answer?"
            ),
            "options": [
                "A) ModelRegistry does not support setting stage at registration time — promote_model is the only way to set stage",
                "B) The two-step pattern enforces a quality gate: register() creates an immutable versioned artifact with metrics (test_accuracy, train_time, n_params), then promote_model() is a deliberate approval action that can be gated by governance rules, automated CI checks, or human review. This separation means you can register many candidate versions and promote only the one meeting your accuracy threshold — crucial in production where a bad promote can break live traffic.",
                "C) promote_model() compresses the model artifact for production — registration stores the full-size version",
                "D) They are equivalent — register with stage='production' and separate register+promote produce identical ModelRegistry records",
            ],
            "answer": "B",
            "explanation": (
                "The ModelRegistry two-step pattern in Exercise 8 reflects production ML lifecycle discipline. "
                "register_model() is idempotent and additive — it captures the artifact with metrics at a "
                "point in time. promote_model() is a deliberate state transition. In production systems "
                "this separation enables: (1) A/B testing — register multiple versions, promote the winner. "
                "(2) Rollback — demote the current production version and promote the previous one. "
                "(3) Governance — an automated check or human approval between registration and promotion. "
                "The capstone intentionally uses this two-step flow to teach the pattern, not just the API. "
                "A model in 'staging' stage can be compared against 'production' before replacing it."
            ),
            "learning_outcome": "Explain the register-then-promote pattern in ModelRegistry for production governance",
        },
        {
            "id": "7.8.3",
            "lesson": "7.8",
            "type": "output_interpret",
            "difficulty": "intermediate",
            "question": (
                "Exercise 8 prints this inference speed comparison output:\n\n"
                "  Original model: 100 predictions in 0.842s (8.4ms/prediction)\n"
                "  ONNX model: typically 2-5x faster due to graph optimizations\n\n"
                "A student replicates the exercise but measures ONNX at 9.1ms/prediction — "
                "SLOWER than the original 8.4ms. They ran the benchmark on a single sample "
                "at a time in a loop. What is the most likely explanation and fix?"
            ),
            "options": [
                "A) The ONNX export failed — the validation step should have caught this discrepancy",
                "B) Single-sample inference benchmarks ONNX Runtime unfairly: ONNX's advantage comes from batched execution and warm JIT compilation. The first inference call incurs session initialization overhead (loading the model, allocating memory, JIT-compiling the graph). Measuring single samples in a loop gives mostly session overhead. Fix: warm up with 10 predictions before benchmarking, then measure in batches of 32+.",
                "C) ONNX Runtime requires a GPU to be faster than Python — on CPU it is always slower",
                "D) The model is too small for ONNX to optimize — ONNX only accelerates models with >10M parameters",
            ],
            "answer": "B",
            "explanation": (
                "ONNX Runtime provides two main speedup sources: (1) Graph-level optimizations applied once "
                "at session creation (operator fusion: Conv+BatchNorm+ReLU → single kernel). "
                "(2) Batched execution — CPU vectorization (SIMD) processes N inputs simultaneously. "
                "Single-sample benchmarks miss both: the optimization overhead is amortized over a session, "
                "not re-applied per call, and SIMD throughput requires a batch > 1. "
                "Exercise 8 is explicit about the benefit: 'typically 2-5x faster due to graph optimizations' "
                "— this refers to production batch inference, not latency-optimized single-sample serving. "
                "A warm-up pass plus batch=32 benchmark would show the expected 2-5x speedup. "
                "This is why InferenceServer handles batching internally — it accumulates requests before "
                "forwarding to the ONNX runtime."
            ),
            "learning_outcome": "Benchmark ONNX inference correctly by accounting for warm-up and batch size",
        },
    ],
}
