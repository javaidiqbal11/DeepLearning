"""Single source of truth for the course.

The slide deck builder, the module PDF and the per-lecture READMEs all read
this file, so the syllabus can never drift between artefacts.

Curriculum design follows the consensus structure of Stanford CS231n (2026),
NYU DS-GA 1008 (LeCun / Canziani) and the deeplearning.ai Deep Learning
Specialization, re-sequenced for a 16-session graduate semester with a
deployment thread (FastAPI) running through it.
"""
from __future__ import annotations

COURSE = {
    "code": "DL-601",
    "title": "Deep Learning",
    "subtitle": "Foundations, Computer Vision and Production Systems",
    "level": "Graduate (Master's / first-year PhD)",
    "credits": "3 credit hours - 16 lectures x 3 hours (2h lecture + 1h lab)",
    "instructor": "Senior AI Engineer, Course Instructor",
    "prerequisites": [
        "Linear algebra: matrices, matrix products, eigenvalues",
        "Multivariable calculus: partial derivatives and the chain rule",
        "Probability: random variables, expectation, maximum likelihood",
        "Python: comfortable with functions, classes, NumPy arrays",
        "No prior deep learning experience assumed",
    ],
    "stack": [
        "Python 3.10+",
        "PyTorch 2.x (CPU is sufficient for every notebook in this course)",
        "NumPy, Matplotlib, scikit-learn, Pillow, pandas",
        "FastAPI + Uvicorn for model serving",
        "Jupyter / JupyterLab",
    ],
    "outcomes": [
        "Explain how a deep network computes its output and how gradients flow back through it.",
        "Implement backpropagation, convolution and attention from first principles in NumPy and PyTorch.",
        "Select and train an appropriate architecture for image classification, detection and segmentation.",
        "Diagnose an underperforming model from its learning curves, confusion matrix and saliency maps.",
        "Apply transfer learning and self-supervised pre-training when labelled data is scarce.",
        "Build and reason about Transformers for both text and image inputs.",
        "Serve a trained model behind a documented, tested FastAPI endpoint.",
        "Evaluate a model for robustness, calibration and fairness before release.",
    ],
    "assessment": [
        ("Weekly lab tasks (16 x 1.5%)", "24%", "Graded on completion and correctness of the core tasks."),
        ("Assignment 1 - Foundations (Lectures 1-4)", "12%", "Backpropagation from scratch and an MLP training study."),
        ("Assignment 2 - Convolutional Vision (Lectures 5-8)", "14%", "CNN design, transfer learning and error analysis."),
        ("Assignment 3 - Sequences and Attention (Lectures 10-14)", "14%", "Detection or segmentation plus a Transformer built from scratch."),
        ("Capstone project (Lectures 15-16)", "26%", "End-to-end system: data, training, evaluation, FastAPI service, report."),
        ("Participation and paper discussion", "10%", "One 10-minute paper presentation per student."),
    ],
    "policies": [
        "Late work: 10% per day, up to three days; after that the task scores zero.",
        "Collaboration: discussion is encouraged, submitted code and text must be your own.",
        "AI assistants: permitted for explanation and debugging, must be disclosed in a short note; "
        "generating an entire solution is not permitted.",
        "Reproducibility: every submission must run end to end from a clean checkout with a fixed seed.",
        "Compute: no GPU is required. If a notebook takes more than ~10 minutes on your laptop, "
        "reduce the epoch count and say so in your write-up.",
    ],
    "grading_scale": [
        ("A", "93-100"), ("A-", "90-92"), ("B+", "87-89"), ("B", "83-86"),
        ("B-", "80-82"), ("C+", "77-79"), ("C", "70-76"), ("F", "below 70"),
    ],
}

PARTS = {
    1: "Part I - Foundations",
    2: "Part II - Convolutional Vision",
    3: "Part III - Vision Systems and Deployment",
    4: "Part IV - Attention, Generation and Production",
}

# --------------------------------------------------------------------------
# Each lecture: the structure every artefact is generated from.
#   objectives  -> learning outcomes, appear on slide 2 and in the PDF
#   outline     -> (section title, [bullet, ...]) driving the body slides
#   lab         -> what the notebook does
#   dataset     -> which generated dataset is used
#   tasks       -> (core | stretch, title, description) for the tasks file
#   reading     -> canonical references
# --------------------------------------------------------------------------
LECTURES: list[dict] = []


def lecture(**kw):
    LECTURES.append(kw)


lecture(
    number=1, part=1,
    title="Deep Learning Foundations and the Image Data Pipeline",
    tagline="Why depth works, and how pixels become tensors.",
    objectives=[
        "Place deep learning relative to classical machine learning and explain what changed after 2012.",
        "Describe an image as a tensor and move fluently between NumPy, PIL and PyTorch layouts.",
        "Build a Dataset and DataLoader and reason about batching, shuffling and normalisation.",
        "Establish a baseline and a train/validation/test protocol before touching a model.",
    ],
    outline=[
        ("What deep learning actually is", [
            "Representation learning: features are learned, not hand-designed.",
            "The classical pipeline (SIFT/HOG + SVM) versus the learned pipeline.",
            "Three enablers: large labelled datasets, GPUs, and better optimisation and initialisation.",
            "AlexNet 2012 cut ImageNet top-5 error from 26% to 15% and ended the hand-crafted-feature era.",
        ]),
        ("When to use deep learning - and when not to", [
            "Good fit: perceptual data (images, audio, text), abundant data, tolerance for opacity.",
            "Poor fit: small tabular datasets, hard interpretability constraints, tight latency on tiny hardware.",
            "Gradient-boosted trees still beat deep nets on most tabular problems. Choose deliberately.",
        ]),
        ("Images as tensors", [
            "A colour image is a rank-3 tensor: (height, width, channels).",
            "PyTorch convention is NCHW; NumPy, PIL and matplotlib use HWC. Most day-one bugs are a permute away.",
            "uint8 in [0, 255] for storage, float32 in [0, 1] for compute, standardised for training.",
            "Normalisation with per-channel mean and standard deviation puts every input on a comparable scale.",
        ]),
        ("The data pipeline", [
            "Dataset defines __len__ and __getitem__; DataLoader adds batching, shuffling and parallel workers.",
            "Shuffle the training set every epoch; never shuffle the test set.",
            "On Windows, use num_workers=0 in notebooks - worker spawning misbehaves inside Jupyter.",
            "Compute normalisation statistics on the training split only. Using all data leaks information.",
        ]),
        ("Splits, baselines and leakage", [
            "Train fits parameters, validation selects hyper-parameters, test is touched exactly once.",
            "Always record the majority-class rate and a linear baseline first.",
            "On our shapes dataset: chance is 25%, a linear model reaches ~63%. That gap is what the course is about.",
            "Leakage checklist: duplicated images across splits, statistics computed on all data, tuning on test.",
        ]),
    ],
    lab="Load the shapes dataset three ways (npz arrays, a Dataset class, an on-disk image folder), "
        "visualise it, compute normalisation statistics correctly, and fit a majority-class and a "
        "linear baseline to fix the numbers the rest of the course must beat.",
    dataset="shapes_32.npz, shapes_imagefolder/",
    tasks=[
        ("core", "Tensor layout drill",
         "Write round_trip(path) that loads a PNG with PIL, converts it to a normalised NCHW float tensor, "
         "converts it back, and asserts the recovered uint8 array matches the original within 1/255. "
         "State in one sentence where the rounding error comes from."),
        ("core", "Custom Dataset",
         "Implement ShapesFolder(Dataset) that reads shapes_imagefolder/train without using ImageFolder. "
         "It must discover class names from the directory listing, sort them for a stable label mapping, "
         "and return (CHW float tensor, int label)."),
        ("core", "Honest normalisation",
         "Compute per-channel mean and std on the training split only. Then compute them on train+val+test "
         "combined. Report both to four decimal places and explain why only the first is admissible."),
        ("core", "Baselines that matter",
         "Report majority-class accuracy and multinomial logistic-regression accuracy on the test split. "
         "Record both in a results table - every later lecture compares against these numbers."),
        ("stretch", "Batch-size timing study",
         "Time one epoch of the linear baseline at batch sizes 1, 16, 128 and 1024. Plot seconds per epoch "
         "against batch size and explain the shape of the curve in terms of Python overhead versus vectorisation."),
        ("stretch", "Find the leak",
         "leaky_split.py builds a train/test split with a deliberate flaw. Find it, quantify how much it "
         "inflates the reported accuracy, and fix it."),
    ],
    reading=[
        "Goodfellow, Bengio & Courville, *Deep Learning*, Chapter 1 and Section 5.1-5.3.",
        "Krizhevsky, Sutskever & Hinton (2012), *ImageNet Classification with Deep CNNs* (AlexNet).",
        "PyTorch docs: Datasets and DataLoaders tutorial.",
    ],
)

lecture(
    number=2, part=1,
    title="From Linear Models to Neural Networks",
    tagline="The softmax classifier, and the exact point where linearity fails.",
    objectives=[
        "Derive the softmax classifier and the cross-entropy loss from maximum likelihood.",
        "Implement the forward pass and analytic gradient of a linear classifier in NumPy.",
        "Explain why stacking linear layers without a non-linearity gains nothing.",
        "Show empirically that a hidden layer plus ReLU solves a problem a linear model cannot.",
    ],
    outline=[
        ("The linear score function", [
            "s = Wx + b maps a flattened image to one score per class.",
            "Each row of W is a learned template; visualising the rows shows what the model looks for.",
            "A linear model can only carve the input space with hyperplanes.",
        ]),
        ("Softmax and cross-entropy", [
            "Softmax turns scores into a probability distribution: p_k = exp(s_k) / sum_j exp(s_j).",
            "Cross-entropy is the negative log-likelihood of the correct class: L = -log p_y.",
            "Subtract max(s) before exponentiating - otherwise exp overflows on large scores.",
            "The gradient is famously clean: dL/ds = p - onehot(y).",
        ]),
        ("Why a non-linearity is required", [
            "W2(W1 x) = (W2 W1) x - composing linear maps yields another linear map.",
            "ReLU(z) = max(0, z) is cheap, non-saturating, and makes the network piecewise linear.",
            "Sigmoid and tanh saturate and kill gradients; ReLU is the default, with GELU common in Transformers.",
            "Dead ReLUs: a unit stuck at zero never recovers. Leaky ReLU trades that for a small negative slope.",
        ]),
        ("The universal approximation theorem, honestly", [
            "One hidden layer of sufficient width can approximate any continuous function on a compact set.",
            "It is an existence result. It says nothing about the width required, or whether SGD will find it.",
            "Depth is exponentially more parameter-efficient than width for many function families.",
        ]),
        ("Loss surfaces and what optimisation faces", [
            "Convex for a linear model with cross-entropy; non-convex the moment a hidden layer appears.",
            "In high dimensions, saddle points - not local minima - are the dominant obstacle.",
            "Most minima found by SGD in over-parameterised nets are of comparable quality.",
        ]),
    ],
    lab="Implement a softmax classifier in pure NumPy with an analytic gradient, verify it against a "
        "numerical gradient, train it on the digits data, then demonstrate the linear/non-linear gap on "
        "a two-moons problem and on the shapes dataset.",
    dataset="digits_8x8.npz, shapes_32.npz",
    tasks=[
        ("core", "Numerically stable softmax",
         "Implement softmax(scores) that handles a row containing 10000 without producing NaN. "
         "Show the naive version overflowing and yours not."),
        ("core", "Analytic gradient plus gradient check",
         "Implement the cross-entropy gradient for a linear classifier. Verify against a central-difference "
         "numerical gradient; relative error must be below 1e-6. Report the worst element."),
        ("core", "Train the linear classifier",
         "Train your NumPy softmax classifier on digits_8x8 with mini-batch gradient descent. Reach at least "
         "90% test accuracy and plot the loss curve."),
        ("core", "Visualise the templates",
         "Reshape each row of the learned W back to 8x8 and plot all ten. Describe in two sentences what the "
         "template for the digit 0 has learned to detect."),
        ("core", "One hidden layer changes everything",
         "On sklearn's two-moons dataset, fit a linear classifier and an MLP with one hidden layer of 32 ReLU "
         "units. Plot both decision boundaries side by side and report both accuracies."),
        ("stretch", "Activation comparison",
         "Train the same MLP on the shapes dataset with ReLU, tanh, sigmoid and LeakyReLU. Plot the four loss "
         "curves on one axis and explain the sigmoid result in terms of gradient saturation."),
        ("stretch", "Depth without non-linearity",
         "Build a five-layer network with no activation functions. Show that its test accuracy matches a single "
         "linear layer, and confirm the product of its weight matrices is a rank-limited linear map."),
    ],
    reading=[
        "Goodfellow et al., *Deep Learning*, Chapter 6.",
        "CS231n course notes: Linear Classification and Optimization.",
        "Nair & Hinton (2010), *Rectified Linear Units Improve Restricted Boltzmann Machines*.",
    ],
)

lecture(
    number=3, part=1,
    title="Backpropagation and Automatic Differentiation",
    tagline="The chain rule, organised as a graph - and built from scratch.",
    objectives=[
        "Derive backpropagation as the reverse-mode application of the chain rule on a computational graph.",
        "Implement a working reverse-mode autodiff engine in under 150 lines of NumPy.",
        "Compute local gradients for the layers used throughout the course.",
        "Recognise vanishing and exploding gradients from the gradient-norm profile across layers.",
    ],
    outline=[
        ("The computational graph", [
            "Every model is a DAG of primitive operations; each node knows its local derivative.",
            "Forward pass computes values and caches whatever the backward pass will need.",
            "Backward pass walks the graph in reverse topological order, multiplying local Jacobians.",
        ]),
        ("Forward mode versus reverse mode", [
            "Forward mode costs one pass per input; reverse mode costs one pass per output.",
            "Deep learning has millions of inputs (parameters) and one output (the loss) - reverse mode wins.",
            "Cost of the backward pass is roughly twice the forward pass, at the price of storing activations.",
        ]),
        ("Local gradients you should know by heart", [
            "Add: distributes the incoming gradient unchanged to both inputs.",
            "Multiply: swaps the inputs - dL/da = b * upstream.",
            "MatMul: dL/dW = upstream^T x, dL/dx = W^T upstream. Check shapes, always.",
            "ReLU: passes the gradient where the input was positive, blocks it elsewhere.",
            "Max/pooling: routes the whole gradient to the argmax, zero everywhere else.",
        ]),
        ("Broadcasting and the sum rule", [
            "Where a tensor was broadcast in the forward pass, sum the gradient over the broadcast axes.",
            "A bias of shape (C,) added to (N, C) accumulates gradient summed over the batch dimension.",
            "Fan-out means a value used twice receives the sum of both gradient paths.",
        ]),
        ("When gradients go wrong", [
            "Vanishing: repeated multiplication by factors below one drives early-layer gradients to zero.",
            "Exploding: factors above one blow up; symptom is a sudden NaN loss. Fix with gradient clipping.",
            "Diagnose by printing the gradient norm per layer - a healthy net keeps them within an order of magnitude.",
            "Residual connections give gradients a path that skips the multiplications entirely.",
        ]),
    ],
    lab="Build a Tensor class with reverse-mode autodiff from scratch (add, mul, matmul, relu, softmax "
        "cross-entropy), validate every gradient against PyTorch, train an MLP with it, then instrument a "
        "deep network to observe vanishing gradients directly.",
    dataset="digits_8x8.npz, shapes_32.npz",
    tasks=[
        ("core", "Hand-derive a graph",
         "For f(x, y, z) = (x + y) * max(z, 0) at x=2, y=-3, z=4, draw the graph, run the forward pass, and "
         "compute all three partial derivatives by hand. Show every intermediate value."),
        ("core", "Finish the autodiff engine",
         "The provided Tensor class has add, mul and matmul. Implement backward for relu, sum, mean and "
         "the log-sum-exp used by cross-entropy."),
        ("core", "Verify against PyTorch",
         "For each operation you implemented, build the same expression in PyTorch with requires_grad=True "
         "and assert the gradients agree to within 1e-6."),
        ("core", "Train with your own engine",
         "Train a two-layer MLP on digits_8x8 using only your engine - no torch.nn, no torch.optim. "
         "Reach at least 92% test accuracy."),
        ("core", "Gradient-norm profile",
         "Build a 15-layer sigmoid MLP. Plot the gradient norm of each layer on a log scale after one backward "
         "pass. Repeat with ReLU and with residual connections; put all three on one figure."),
        ("stretch", "Add Conv2d",
         "Extend your engine with a 2D convolution and its backward pass. Verify against "
         "torch.nn.functional.conv2d on a random 4x3x8x8 input."),
        ("stretch", "Gradient checkpointing",
         "Modify the engine to recompute activations instead of storing them. Measure peak memory and wall-clock "
         "time for both versions and report the trade-off."),
    ],
    reading=[
        "Goodfellow et al., *Deep Learning*, Section 6.5.",
        "Baydin et al. (2018), *Automatic Differentiation in Machine Learning: a Survey*.",
        "Karpathy, *micrograd* - a 150-line autodiff engine worth reading line by line.",
    ],
)

lecture(
    number=4, part=1,
    title="Training Deep Networks: Optimisation and Regularisation",
    tagline="Everything between 'it runs' and 'it works'.",
    objectives=[
        "Compare SGD, momentum, RMSProp and Adam, and state when each is the right default.",
        "Choose an initialisation scheme that keeps activation variance stable with depth.",
        "Apply batch normalisation, dropout and weight decay, and explain what each actually does.",
        "Diagnose underfitting and overfitting from learning curves and respond correctly.",
    ],
    outline=[
        ("From gradient descent to Adam", [
            "SGD: cheap, noisy, and the noise itself is a useful regulariser.",
            "Momentum accumulates a velocity vector and damps oscillation across ravines.",
            "RMSProp scales each coordinate by a running estimate of its gradient magnitude.",
            "Adam = momentum + RMSProp + bias correction. Defaults: lr 1e-3, betas (0.9, 0.999).",
            "AdamW decouples weight decay from the gradient update and is the correct choice for Transformers.",
        ]),
        ("Learning rate: the hyper-parameter that matters most", [
            "Too high diverges; too low crawls or stalls in a bad region.",
            "LR range test: sweep the rate exponentially for one epoch and plot loss against rate.",
            "Schedules: step, cosine annealing, and linear warmup for large batches or Transformers.",
            "Warmup exists because early Adam updates are badly scaled before the variance estimates settle.",
        ]),
        ("Initialisation", [
            "All zeros breaks symmetry permanently - every unit computes the same thing forever.",
            "Xavier/Glorot: variance 1/fan_in, derived for tanh.",
            "He/Kaiming: variance 2/fan_in, corrects for ReLU zeroing half the activations. Use this with ReLU.",
            "Bad initialisation looks exactly like a bad learning rate. Check it first.",
        ]),
        ("Normalisation layers", [
            "BatchNorm standardises each channel over the batch, then applies a learned scale and shift.",
            "It permits higher learning rates and reduces sensitivity to initialisation.",
            "Train and eval behave differently - forgetting model.eval() is a classic production bug.",
            "LayerNorm normalises over features per sample, is batch-size independent, and is standard in Transformers.",
        ]),
        ("Regularisation", [
            "L2 / weight decay pulls weights toward zero and prefers smoother functions.",
            "Dropout randomly zeroes units during training, approximating an ensemble of subnetworks.",
            "Early stopping on validation loss is the cheapest regulariser available.",
            "Data augmentation is usually worth more than any explicit penalty - see Lecture 7.",
        ]),
        ("Reading the learning curves", [
            "High train loss and high val loss: underfitting - more capacity, longer training, higher LR.",
            "Low train loss and high val loss: overfitting - more data, augmentation, regularisation.",
            "Val loss rising while val accuracy holds: the model is growing over-confident. Check calibration.",
            "Erratic loss: learning rate too high, or a batch size of one on BatchNorm.",
        ]),
    ],
    lab="Run controlled experiments: an optimiser bake-off on identical seeds, a learning-rate range test, "
        "an initialisation comparison measured by activation variance per layer, and an ablation of BatchNorm, "
        "dropout and weight decay on a deliberately overfitting setup.",
    dataset="shapes_32.npz",
    tasks=[
        ("core", "Optimiser bake-off",
         "Train the same MLP on shapes with SGD, SGD+momentum, RMSProp and Adam. Fix the seed and every other "
         "hyper-parameter. Plot four validation-loss curves on one axis and name the winner at 20 epochs."),
        ("core", "Learning-rate range test",
         "Sweep the learning rate from 1e-5 to 1e0 over one epoch, recording the loss after each batch. "
         "Plot loss against log learning rate and identify the largest rate that is still stable."),
        ("core", "Initialisation and activation variance",
         "For a 10-layer ReLU MLP initialised with zeros, N(0, 0.01), Xavier and He, plot the variance of each "
         "layer's activations. Explain why He keeps it flat."),
        ("core", "Regularisation ablation",
         "Take a model that overfits (train 99%, val 78%). Add weight decay, dropout, and both. Report a four-row "
         "table of train and validation accuracy and state which intervention paid off."),
        ("core", "BatchNorm in train versus eval",
         "Train a small CNN with BatchNorm. Evaluate the test set once in train() mode and once in eval() mode. "
         "Report both numbers and explain the difference in terms of batch statistics versus running statistics."),
        ("stretch", "Cosine schedule with warmup",
         "Implement cosine annealing with linear warmup as a LambdaLR. Plot the rate over 50 epochs and compare "
         "final accuracy against a constant rate."),
        ("stretch", "Batch size and generalisation",
         "Train at batch sizes 8, 64, 512 and 4096 with the learning rate scaled linearly. Report final validation "
         "accuracy and comment on the large-batch generalisation gap."),
    ],
    reading=[
        "Kingma & Ba (2015), *Adam: A Method for Stochastic Optimization*.",
        "He et al. (2015), *Delving Deep into Rectifiers* (He initialisation).",
        "Ioffe & Szegedy (2015), *Batch Normalization*.",
        "Loshchilov & Hutter (2019), *Decoupled Weight Decay Regularization* (AdamW).",
    ],
)

lecture(
    number=5, part=2,
    title="Convolutional Neural Networks",
    tagline="The right inductive bias for images, derived rather than asserted.",
    objectives=[
        "Explain convolution as local connectivity plus weight sharing, and count the parameters it saves.",
        "Compute output shapes for any kernel, stride, padding and dilation without guessing.",
        "Implement 2D convolution and max pooling from scratch, then match PyTorch's result.",
        "Reason about receptive field growth and design a CNN whose receptive field covers the input.",
    ],
    outline=[
        ("Why not just use an MLP", [
            "A 224x224x3 image into one 1000-unit hidden layer is 150 million weights in a single layer.",
            "An MLP discards spatial structure the moment it flattens the input.",
            "It has no translation equivariance: shift the object by one pixel and every input changes.",
        ]),
        ("The two ideas", [
            "Local connectivity: a unit sees a small patch, because pixels far apart are weakly related.",
            "Weight sharing: the same kernel slides everywhere, because an edge is an edge anywhere in the frame.",
            "Together they give translation equivariance and cut parameters by orders of magnitude.",
            "A 3x3x3 kernel producing 64 channels is 1792 parameters, independent of image size.",
        ]),
        ("Convolution arithmetic", [
            "out = floor((in + 2*padding - dilation*(kernel - 1) - 1) / stride) + 1.",
            "'same' padding for an odd kernel k with stride 1 is p = (k - 1) / 2.",
            "Stride 2 halves the spatial size and is the common alternative to pooling.",
            "Dilation enlarges the receptive field without adding parameters - central to segmentation.",
        ]),
        ("Pooling and downsampling", [
            "Max pooling gives small-translation invariance and reduces computation.",
            "Average pooling is smoother; global average pooling replaces the big flatten-and-dense head.",
            "Modern architectures often use strided convolution instead of pooling and learn the downsample.",
        ]),
        ("Receptive field", [
            "Stacking two 3x3 layers gives a 5x5 receptive field with fewer parameters than one 5x5 layer.",
            "RF grows linearly with depth and multiplicatively with stride.",
            "If the receptive field at the classification layer does not cover the object, the model cannot see it.",
        ]),
        ("What the filters learn", [
            "Layer 1: oriented edges and colour blobs - remarkably close to Gabor filters, every time.",
            "Middle layers: corners, textures, repeated motifs.",
            "Deep layers: object parts and whole-object detectors.",
            "This hierarchy emerges from the data. Nobody designs it.",
        ]),
    ],
    lab="Implement conv2d with im2col and max pooling from scratch and match torch.nn.functional to 1e-5; "
        "apply hand-built edge kernels to see what convolution does; train a CNN to ~99% and compare it "
        "against an MLP on accuracy, parameter count and — the decisive test — accuracy on translated "
        "images; then visualise the learned first-layer filters and measure the receptive field.",
    dataset="shapes_32.npz",
    tasks=[
        ("core", "Shape arithmetic without running code",
         "For a 32x32x3 input, compute the output shape and parameter count for: Conv2d(3,16,3,pad=1); "
         "MaxPool2d(2); Conv2d(16,32,5,stride=2,pad=2); Conv2d(32,32,3,dilation=2,pad=2). "
         "Then verify each with a forward pass."),
        ("core", "Convolution from scratch",
         "Implement conv2d_naive(x, w, b, stride, padding) with explicit loops, and conv2d_im2col using matrix "
         "multiplication. Both must match F.conv2d to within 1e-5. Report the speed-up of im2col."),
        ("core", "Max pooling forward and backward",
         "Implement max pooling and its backward pass. Verify the gradient routes entirely to the argmax position."),
        ("core", "Train a CNN on shapes",
         "Build a three-block CNN (conv-ReLU-pool) with a dense head. Train to at least 97% test accuracy and "
         "add the result to your running results table beside the Lecture 1 baselines."),
        ("core", "Parameter accounting",
         "Compare your CNN against an MLP with the same test accuracy. Report parameter counts for both and "
         "explain the ratio in terms of weight sharing."),
        ("core", "Visualise layer-1 filters",
         "Plot all first-layer kernels as RGB images. Identify at least two that respond to oriented edges and "
         "show their activation maps on one test image."),
        ("stretch", "Receptive field calculator",
         "Write receptive_field(layers) that returns the RF size at each layer for a list of "
         "(kernel, stride, dilation) tuples. Verify empirically by finding which input pixels affect one output unit."),
        ("stretch", "Translation equivariance test",
         "Shift a test image by 1, 4 and 8 pixels. Measure how the CNN's and the MLP's predictions change, "
         "and quantify the difference."),
    ],
    reading=[
        "LeCun et al. (1998), *Gradient-Based Learning Applied to Document Recognition* (LeNet-5).",
        "Dumoulin & Visin (2016), *A Guide to Convolution Arithmetic for Deep Learning*.",
        "CS231n notes: Convolutional Neural Networks.",
    ],
)

lecture(
    number=6, part=2,
    title="Modern CNN Architectures and Transfer Learning",
    tagline="ResNet, and why fine-tuning beats training from scratch almost every time.",
    objectives=[
        "Trace the architectural lineage from LeNet through AlexNet, VGG and Inception to ResNet.",
        "Explain the degradation problem and how residual connections resolve it.",
        "Implement a residual block and a small ResNet from scratch.",
        "Choose correctly between feature extraction and fine-tuning given dataset size and domain distance.",
    ],
    outline=[
        ("The architecture lineage", [
            "LeNet-5 (1998): the template - conv, pool, conv, pool, dense.",
            "AlexNet (2012): ReLU, dropout, GPUs, augmentation. The result that restarted the field.",
            "VGG (2014): only 3x3 convolutions, uniform and deep. Enormous - 138M parameters.",
            "Inception/GoogLeNet (2014): parallel multi-scale branches, 1x1 convolutions for cheap channel mixing.",
            "ResNet (2015): residual connections made 152 layers trainable and won ImageNet at 3.57% top-5 error.",
        ]),
        ("The degradation problem", [
            "A 56-layer plain network had higher *training* error than a 20-layer one. Not overfitting - optimisation.",
            "Deeper networks should be at least as good: the extra layers could learn the identity. They did not.",
            "Residual block: y = F(x) + x. Learning F = 0 is easy, so the identity is the default behaviour.",
            "The skip connection also gives gradients an unobstructed path back to the early layers.",
        ]),
        ("Building blocks worth knowing", [
            "1x1 convolution: channel mixing and dimensionality reduction at negligible spatial cost.",
            "Bottleneck block: 1x1 reduce, 3x3 process, 1x1 expand - the workhorse of ResNet-50 and deeper.",
            "Depthwise separable convolution: MobileNet's trick, 8-9x cheaper than standard convolution.",
            "Squeeze-and-excitation: learned per-channel attention, a cheap and reliable accuracy gain.",
        ]),
        ("Transfer learning", [
            "Early layers learn generic features (edges, textures) that transfer across almost any image domain.",
            "Late layers are task-specific and are the ones to replace.",
            "Feature extraction: freeze the backbone, train a new head. Right for small or similar datasets.",
            "Fine-tuning: unfreeze some or all layers at a low learning rate (typically 10x lower).",
            "Discriminative learning rates: lower for early layers, higher for late ones.",
        ]),
        ("Choosing a strategy", [
            "Small data, similar domain: freeze the backbone, train a linear head.",
            "Large data, similar domain: fine-tune everything.",
            "Small data, distant domain: fine-tune the middle layers; the very late features may not transfer.",
            "Large data, distant domain: fine-tuning still usually beats random initialisation on convergence speed.",
        ]),
    ],
    lab="Implement a residual block and a small ResNet, empirically reproduce the degradation problem with a "
        "20- versus 40-layer plain network, then run the transfer-learning comparison: a source model "
        "pre-trained on one shape subset, transferred to a 100-example target task, versus training from scratch.",
    dataset="shapes_32.npz, shapes_imagefolder/",
    tasks=[
        ("core", "Residual block from scratch",
         "Implement BasicBlock with two 3x3 convolutions, BatchNorm and a skip connection, including the 1x1 "
         "projection needed when channel count or stride changes. Verify output shapes for both cases."),
        ("core", "Reproduce the degradation problem",
         "Train a 20-layer and a 40-layer plain CNN. Show the deeper one has higher *training* loss. "
         "Add skip connections to both and show the ordering reverses."),
        ("core", "Small ResNet on shapes",
         "Assemble a ResNet-style network from your blocks. Reach at least 99% test accuracy and report "
         "parameter count and training time against the Lecture 5 CNN."),
        ("core", "Feature extraction versus fine-tuning",
         "Pre-train on circle/square only. Transfer to a triangle/star task with just 100 labelled examples, "
         "three ways: from scratch, frozen backbone, full fine-tuning at lr/10. Report all three accuracies."),
        ("core", "How many layers to unfreeze",
         "Sweep the number of unfrozen backbone blocks from 0 to all. Plot target-task accuracy against that "
         "number and state where the curve flattens."),
        ("stretch", "1x1 bottleneck cost analysis",
         "Compare parameters and FLOPs for a plain 3x3-3x3 block against a 1x1-3x3-1x1 bottleneck of equal "
         "input/output width. Report the ratio and verify with a forward-pass timing."),
        ("stretch", "Depthwise separable convolution",
         "Implement it and substitute it into your ResNet. Report the accuracy lost and the parameters saved."),
    ],
    reading=[
        "He et al. (2016), *Deep Residual Learning for Image Recognition*.",
        "Simonyan & Zisserman (2015), *Very Deep Convolutional Networks* (VGG).",
        "Yosinski et al. (2014), *How Transferable Are Features in Deep Neural Networks?*",
        "Howard et al. (2017), *MobileNets*.",
    ],
)

lecture(
    number=7, part=2,
    title="Data Augmentation, Training Recipes and Experiment Tracking",
    tagline="The unglamorous work that produces most of the accuracy.",
    objectives=[
        "Design an augmentation policy whose transformations preserve the label.",
        "Implement mixup and cutmix and explain what they regularise.",
        "Run a reproducible experiment: fixed seeds, logged configuration, recorded artefacts.",
        "Choose between grid, random and successive-halving hyper-parameter search.",
    ],
    outline=[
        ("Augmentation as a prior", [
            "Augmentation encodes the invariances you know the task has.",
            "It is the cheapest way to buy effective data, and usually outperforms explicit regularisers.",
            "The rule: the transformation must not change the correct label.",
            "A counter-example: vertical flip changes a 6 into a 9. Never flip digits vertically.",
        ]),
        ("The standard geometric and photometric toolkit", [
            "Random crop with padding, horizontal flip, small rotation, scale and translate.",
            "Colour jitter (brightness, contrast, saturation, hue), grayscale, Gaussian blur.",
            "Random erasing / cutout: occlude a rectangle so the model cannot rely on a single region.",
            "Augment the training set only. Augmenting validation makes your metric meaningless.",
        ]),
        ("Mixup and cutmix", [
            "Mixup: x = lam*x_i + (1-lam)*x_j with the labels mixed identically, lam ~ Beta(a, a).",
            "It encourages linear behaviour between examples and improves calibration noticeably.",
            "Cutmix pastes a patch of one image into another, mixing labels by patch area.",
            "Both need soft-label cross-entropy, and both usually want longer training.",
        ]),
        ("Test-time augmentation", [
            "Average predictions over several augmented copies of the test image.",
            "Reliably worth a point or two of accuracy at a linear cost in inference time.",
            "State clearly whether a reported number used TTA - otherwise the comparison is dishonest.",
        ]),
        ("Reproducibility in practice", [
            "Seed Python, NumPy and PyTorch; set deterministic algorithms when you need bitwise repeatability.",
            "Log the full configuration, the git commit, the library versions and the hardware.",
            "Save the config next to the checkpoint. A checkpoint without its config is nearly worthless.",
            "GPU non-determinism is real: expect small run-to-run variation even with seeds fixed.",
        ]),
        ("Hyper-parameter search", [
            "Grid search wastes evaluations on parameters that do not matter.",
            "Random search covers important dimensions far better at the same budget (Bergstra & Bengio).",
            "Successive halving / Hyperband: start many runs, kill the weak ones early.",
            "Search on validation. Report on test, once.",
        ]),
    ],
    lab="Build an augmentation pipeline from tensor operations, measure each transformation's effect in a "
        "controlled ablation, implement mixup and cutmix with soft-label loss, then build a small experiment "
        "tracker (JSON log + results table) and run a random search over the training recipe.",
    dataset="shapes_32.npz",
    tasks=[
        ("core", "Augmentation from scratch",
         "Implement random_crop_with_padding, random_horizontal_flip, random_rotation and colour_jitter as "
         "functions on CHW float tensors. Show a grid of 16 augmented copies of one image."),
        ("core", "Label-preserving audit",
         "For each of your transformations, state whether it preserves the label for the shapes dataset and "
         "justify it. Identify one transformation that would break the label and explain why."),
        ("core", "Ablation study",
         "Train with no augmentation, then adding one transformation at a time. Produce a table of validation "
         "accuracy and identify which transformation contributes most."),
        ("core", "Mixup",
         "Implement mixup with Beta(0.2, 0.2) and the corresponding soft-label cross-entropy. Train with it and "
         "report the change in validation accuracy and in expected calibration error."),
        ("core", "Experiment tracker",
         "Write a Tracker class that records config, per-epoch metrics, git commit and library versions to a JSON "
         "file per run, and a function that loads all runs into a sorted results DataFrame."),
        ("core", "Random search",
         "Run 12 random-search trials over learning rate (log-uniform 1e-4 to 1e-1), weight decay, dropout and "
         "augmentation strength. Report the best configuration and its validation accuracy."),
        ("stretch", "Cutmix",
         "Implement cutmix, compare it against mixup at equal epoch budget, and show example mixed images with "
         "their soft labels."),
        ("stretch", "Test-time augmentation",
         "Average predictions over 8 augmented copies at test time. Report the accuracy gain and the added latency "
         "per image in milliseconds."),
    ],
    reading=[
        "Zhang et al. (2018), *mixup: Beyond Empirical Risk Minimization*.",
        "Yun et al. (2019), *CutMix*.",
        "Bergstra & Bengio (2012), *Random Search for Hyper-Parameter Optimization*.",
        "Shorten & Khoshgoftaar (2019), *A Survey on Image Data Augmentation*.",
    ],
)

lecture(
    number=8, part=2,
    title="Evaluation, Interpretability and Model Debugging",
    tagline="Accuracy is one number. It is rarely the one you need.",
    objectives=[
        "Select metrics appropriate to class imbalance and to the cost structure of the errors.",
        "Assess calibration and correct it with temperature scaling.",
        "Implement saliency maps and Grad-CAM and read them critically.",
        "Follow a systematic debugging protocol instead of changing hyper-parameters at random.",
    ],
    outline=[
        ("Beyond accuracy", [
            "With 99% negatives, a model predicting 'negative' always scores 99%. Accuracy is worthless there.",
            "Precision, recall, F1; macro-average treats classes equally, micro-average weights by frequency.",
            "ROC-AUC is threshold-free; PR-AUC is the honest choice under heavy imbalance.",
            "Top-5 accuracy matters when classes are genuinely ambiguous.",
        ]),
        ("The confusion matrix is the first diagnostic", [
            "It converts one number into a map of which classes are being confused for which.",
            "Systematic off-diagonal mass points at a labelling problem or a genuine visual ambiguity.",
            "Always inspect the highest-confidence errors by eye. They are the most informative images you have.",
        ]),
        ("Calibration", [
            "A well-calibrated model that says 0.8 is right 80% of the time. Modern networks are badly over-confident.",
            "Reliability diagram: bin by confidence, plot accuracy against mean confidence per bin.",
            "Expected calibration error summarises the gap in a single number.",
            "Temperature scaling divides logits by a single learned T tuned on validation. Simple and effective.",
        ]),
        ("Interpretability methods", [
            "Vanilla saliency: the gradient of the class score with respect to the input pixels. Noisy but instant.",
            "Grad-CAM: weight the last conv feature maps by their gradients; coarse, class-discriminative, robust.",
            "Occlusion sensitivity: slide a grey patch and watch the score drop. Slow, model-agnostic, trustworthy.",
            "Integrated gradients satisfies useful axioms that raw gradients do not.",
            "Caveat: saliency maps have failed sanity checks - some look plausible on a randomly initialised network.",
        ]),
        ("Shortcut learning", [
            "Models exploit whatever correlates with the label, including artefacts you never intended.",
            "Classic failures: hospital tokens in X-rays, snow in husky images, watermarks in scraped data.",
            "Grad-CAM outside the object is the signature. Test on data where the shortcut is broken.",
        ]),
        ("A debugging protocol", [
            "1. Overfit a single batch. If the loss will not reach zero, the bug is in the model or the loss.",
            "2. Check the initial loss equals -log(1/num_classes) for a balanced problem.",
            "3. Verify shapes, label alignment and the normalisation applied at train and at inference.",
            "4. Only then tune the learning rate, the capacity and the regularisation - in that order.",
        ]),
    ],
    lab="Build a full evaluation report for the Lecture 6 model: per-class metrics, confusion matrix, "
        "highest-confidence errors, reliability diagram with temperature scaling, saliency and Grad-CAM. "
        "Then diagnose a deliberately shortcut-corrupted dataset and prove the shortcut with Grad-CAM.",
    dataset="shapes_32.npz, shapes_det.npz",
    tasks=[
        ("core", "Full metric report",
         "Compute per-class precision, recall and F1, plus macro and micro averages, without sklearn.metrics. "
         "Verify your numbers against sklearn afterwards."),
        ("core", "Error gallery",
         "Plot the 16 misclassified test images with the highest predicted confidence, captioned with true and "
         "predicted labels. Write three sentences on what they have in common."),
        ("core", "Reliability diagram and ECE",
         "Implement a 10-bin reliability diagram and expected calibration error. Report the ECE of your model."),
        ("core", "Temperature scaling",
         "Fit a single temperature on the validation set by minimising NLL. Report ECE before and after, and "
         "confirm accuracy is unchanged."),
        ("core", "Grad-CAM",
         "Implement Grad-CAM using forward and backward hooks on the last convolutional block. Produce overlays "
         "for one correct and one incorrect prediction per class."),
        ("core", "Catch the shortcut",
         "shortcut_data.py adds a small class-correlated marker to the corner of each training image. Train on it, "
         "observe the high validation accuracy, then use Grad-CAM to prove the model is reading the marker. "
         "Report accuracy on the clean test set."),
        ("stretch", "Occlusion sensitivity",
         "Implement occlusion sensitivity with a sliding grey patch. Compare its map against Grad-CAM on the same "
         "image and comment on where they disagree."),
        ("stretch", "Saliency sanity check",
         "Reproduce the Adebayo et al. model-randomisation test: compare saliency from a trained model against a "
         "randomly initialised one. Report whether your method passes."),
    ],
    reading=[
        "Guo et al. (2017), *On Calibration of Modern Neural Networks*.",
        "Selvaraju et al. (2017), *Grad-CAM*.",
        "Adebayo et al. (2018), *Sanity Checks for Saliency Maps*.",
        "Geirhos et al. (2020), *Shortcut Learning in Deep Neural Networks*.",
    ],
)

lecture(
    number=9, part=3,
    title="Serving Deep Learning Models with FastAPI",
    tagline="A model that nobody can call is a model that does not exist.",
    objectives=[
        "Export a trained model and load it for inference with correct, reproducible preprocessing.",
        "Build a FastAPI service that accepts an image upload and returns calibrated class probabilities.",
        "Validate inputs with Pydantic and return useful, correct HTTP error codes.",
        "Measure latency and throughput, and improve both with batching and threading controls.",
    ],
    outline=[
        ("Training code is not serving code", [
            "Training optimises throughput on batches; serving optimises latency on single requests.",
            "The preprocessing at inference must match training exactly - the commonest silent production bug.",
            "Ship the preprocessing with the weights. A checkpoint alone is not a deployable artefact.",
        ]),
        ("Model export formats", [
            "state_dict: the recommended PyTorch format, but it requires the class definition to load.",
            "TorchScript (torch.jit.script / trace): self-contained, runs without the Python class.",
            "ONNX: cross-runtime, good for deployment outside Python.",
            "Pickling a whole model couples you to your directory layout. Avoid it.",
        ]),
        ("FastAPI essentials", [
            "Path operations, type hints and automatic OpenAPI documentation at /docs.",
            "Pydantic models validate and coerce request and response bodies for free.",
            "UploadFile streams a file without loading it all into memory.",
            "Load the model once at startup with a lifespan handler, never per request.",
        ]),
        ("Correctness at the edges", [
            "Reject non-image uploads with 415, oversized files with 413, malformed input with 422.",
            "Set torch.set_num_threads deliberately - thread oversubscription is a common latency cliff.",
            "Always wrap inference in torch.inference_mode(). Forgetting it leaks memory and time.",
            "Return the model version alongside every prediction so results are traceable.",
        ]),
        ("Performance", [
            "Measure p50, p95 and p99 latency, not the mean. Tail latency is what users feel.",
            "Dynamic batching: collect requests for a few milliseconds and run them together.",
            "Warm up the model at startup - the first inference is always slower.",
            "Health endpoint for orchestration, readiness distinct from liveness.",
        ]),
        ("Testing a model service", [
            "TestClient gives you fast endpoint tests with no network.",
            "Test the contract (status codes, schema) separately from the model quality.",
            "Golden-output test: a fixed input must produce the same prediction after any refactor.",
        ]),
    ],
    lab="Take the Lecture 6 ResNet, export it with its preprocessing config, build a complete FastAPI service "
        "(upload, batch, health, metrics endpoints), write pytest tests with TestClient, and run a latency "
        "benchmark reporting p50/p95/p99 before and after batching.",
    dataset="shapes_32.npz, shapes_imagefolder/",
    app="Complete FastAPI service in app/ - main.py, model.py, schemas.py, tests/, requirements.txt",
    tasks=[
        ("core", "Export a deployable artefact",
         "Save a checkpoint containing the state_dict, the class names, the normalisation statistics, the input "
         "size and a version string. Write load_model(path) that reconstructs the model and its preprocessing."),
        ("core", "The predict endpoint",
         "Implement POST /predict accepting an image upload and returning JSON with predicted class, all class "
         "probabilities, model version and inference time in milliseconds."),
        ("core", "Input validation",
         "Return 415 for a non-image content type, 413 for a file over 5 MB, and 422 for a corrupt image. "
         "Write a test for each case."),
        ("core", "Batch endpoint",
         "Implement POST /predict/batch accepting multiple files and running them as a single forward pass. "
         "Show it is faster per image than looping over /predict."),
        ("core", "Test suite",
         "Write at least six pytest tests with TestClient covering health, a successful prediction, each error "
         "case, and a golden-output test that pins one fixed image to one prediction."),
        ("core", "Latency benchmark",
         "Measure p50, p95 and p99 latency over 200 single-image requests. Repeat with the batch endpoint at "
         "batch size 16 and report per-image latency for both."),
        ("stretch", "TorchScript comparison",
         "Export the model with torch.jit.script and serve it. Compare cold-start time and p95 latency against "
         "the state_dict version."),
        ("stretch", "Containerise",
         "Write a Dockerfile using a slim Python base and a multi-stage build. Report the final image size and "
         "the container cold-start time."),
    ],
    reading=[
        "FastAPI documentation: Request Files, Dependencies, Testing, Lifespan Events.",
        "PyTorch documentation: Saving and Loading Models; TorchScript.",
        "Sculley et al. (2015), *Hidden Technical Debt in Machine Learning Systems*.",
    ],
)

lecture(
    number=10, part=3,
    title="Object Detection",
    tagline="From 'what is in this image' to 'what, and exactly where'.",
    objectives=[
        "Define the detection task and the role of IoU, non-maximum suppression and mean average precision.",
        "Contrast two-stage (R-CNN family) with one-stage (YOLO, SSD, RetinaNet) detectors.",
        "Implement a single-scale anchor-free detector with classification and box-regression heads.",
        "Compute mAP correctly and interpret a precision-recall curve.",
    ],
    outline=[
        ("The task and why it is harder", [
            "Variable number of outputs per image - a classification head cannot express that.",
            "Objects appear at many scales and positions, and they overlap.",
            "Extreme foreground/background imbalance: most locations contain nothing.",
        ]),
        ("Intersection over Union", [
            "IoU = area of overlap / area of union. The universal matching criterion.",
            "A prediction counts as a true positive at IoU >= threshold, conventionally 0.5.",
            "COCO-style mAP averages over IoU thresholds 0.5 to 0.95 in steps of 0.05 - a much stricter metric.",
        ]),
        ("Two-stage detectors", [
            "R-CNN: region proposals, then a CNN per region. Accurate and extremely slow.",
            "Fast R-CNN: one CNN pass over the image, RoI pooling per proposal.",
            "Faster R-CNN: the Region Proposal Network makes proposals learnable and end-to-end.",
            "Mask R-CNN adds a mask head and RoIAlign, fixing RoI pooling's quantisation error.",
        ]),
        ("One-stage detectors", [
            "YOLO: a single pass predicts boxes and classes on a grid. Fast enough for video.",
            "SSD: predictions from multiple feature-map scales.",
            "RetinaNet: focal loss down-weights easy negatives and closes most of the accuracy gap.",
            "Anchor-free (FCOS, CenterNet): predict centre and extent directly, no anchor tuning.",
            "DETR: set prediction with a Transformer and bipartite matching - no NMS at all.",
        ]),
        ("Non-maximum suppression", [
            "Detectors emit many overlapping boxes for one object.",
            "NMS: sort by score, keep the best, delete anything overlapping it above the IoU threshold, repeat.",
            "Class-wise NMS avoids suppressing a genuinely overlapping object of a different class.",
            "Soft-NMS decays scores instead of deleting, which helps in crowded scenes.",
        ]),
        ("Losses and evaluation", [
            "Classification: cross-entropy or focal loss, over grid locations.",
            "Box regression: smooth L1, or IoU-family losses (GIoU, DIoU, CIoU) which optimise the metric directly.",
            "mAP: average precision per class, averaged over classes. Read the PR curve, not only the summary number.",
        ]),
    ],
    lab="Implement IoU and NMS from scratch with tests, build a single-scale anchor-free detector (a centre "
        "heatmap plus a size regression head) on the 96x96 multi-shape scenes, train it, decode its output, "
        "and implement mAP@0.5 to evaluate it properly.",
    dataset="shapes_det.npz, shapes_det_annotations.json",
    tasks=[
        ("core", "IoU, vectorised",
         "Implement box_iou(boxes_a, boxes_b) returning the full pairwise matrix. Include tests for identical "
         "boxes (1.0), disjoint boxes (0.0), and one box contained in another."),
        ("core", "Non-maximum suppression",
         "Implement NMS from scratch and verify it against torchvision.ops.nms if available, otherwise against "
         "a hand-worked example of five overlapping boxes."),
        ("core", "Build the detector",
         "Implement a CNN backbone with two heads: a centre heatmap over a 24x24 grid and a 2-channel width/height "
         "regression. Train with focal loss on the heatmap and L1 on the sizes."),
        ("core", "Decode predictions",
         "Write decode(heatmap, sizes, threshold) that extracts peaks, converts grid coordinates to pixel boxes, "
         "and applies NMS. Visualise predictions against ground truth for eight validation scenes."),
        ("core", "mean Average Precision",
         "Implement AP at IoU 0.5 using the all-point interpolation rule, then mAP over the four classes. "
         "Plot the precision-recall curve for each class."),
        ("core", "Confidence threshold sweep",
         "Sweep the score threshold from 0.05 to 0.95. Plot precision and recall against threshold and pick an "
         "operating point, justifying it for a named use case."),
        ("stretch", "Focal loss ablation",
         "Train with plain cross-entropy instead of focal loss. Report the mAP difference and explain it via the "
         "foreground/background ratio in your data."),
        ("stretch", "Multi-scale predictions",
         "Add a second prediction head at a coarser stride. Report mAP separately for small and large objects."),
    ],
    reading=[
        "Ren et al. (2015), *Faster R-CNN*.",
        "Redmon et al. (2016), *You Only Look Once*.",
        "Lin et al. (2017), *Focal Loss for Dense Object Detection*.",
        "Tian et al. (2019), *FCOS: Fully Convolutional One-Stage Object Detection*.",
    ],
)

lecture(
    number=11, part=3,
    title="Semantic Segmentation and Dense Prediction",
    tagline="One prediction per pixel, and the encoder-decoder that makes it possible.",
    objectives=[
        "Distinguish semantic, instance and panoptic segmentation and their evaluation metrics.",
        "Explain why an encoder-decoder with skip connections outperforms naive upsampling.",
        "Implement U-Net from scratch and train it on multi-object scenes.",
        "Compute mean IoU and Dice, and apply loss functions that handle class imbalance.",
    ],
    outline=[
        ("Three segmentation tasks", [
            "Semantic: one class label per pixel; two adjacent cars are one 'car' region.",
            "Instance: separates object instances, but ignores background 'stuff'.",
            "Panoptic: unifies both - every pixel gets a class and, where applicable, an instance id.",
        ]),
        ("The resolution problem", [
            "Classification backbones downsample aggressively - 32x is typical - to build semantic depth.",
            "Dense prediction needs full input resolution back.",
            "Naive upsampling from a 1/32-scale map produces blobby, boundary-blind masks.",
        ]),
        ("Encoder-decoder with skip connections", [
            "Encoder: downsample, growing the receptive field and the semantic content.",
            "Decoder: upsample back to input resolution.",
            "Skip connections carry high-resolution spatial detail from encoder to decoder.",
            "U-Net's symmetric design with concatenating skips remains the strongest simple baseline.",
        ]),
        ("Upsampling operators", [
            "Nearest and bilinear interpolation: parameter-free, no checkerboard artefacts.",
            "Transposed convolution: learned, but produces checkerboard artefacts when stride and kernel mismatch.",
            "Recommended: bilinear upsample followed by a 3x3 convolution. Cleaner and just as expressive.",
            "Dilated/atrous convolution: keep resolution while growing the receptive field (DeepLab).",
        ]),
        ("Losses for dense prediction", [
            "Pixel-wise cross-entropy is the default but is dominated by background pixels.",
            "Class-weighted cross-entropy: weight inversely to pixel frequency.",
            "Dice loss optimises overlap directly and is robust to imbalance.",
            "Combined CE + Dice is the reliable practical default.",
        ]),
        ("Evaluation", [
            "Per-class IoU = TP / (TP + FP + FN), then mean IoU over classes.",
            "Dice = 2TP / (2TP + FP + FN) - monotonically related to IoU but weights differently.",
            "Report per-class IoU, never only the mean. The mean hides a failed rare class.",
            "Pixel accuracy is misleading: predicting all background can score 90%.",
        ]),
    ],
    lab="Implement U-Net from scratch, train it on the 96x96 shape scenes for 5-class segmentation, compare "
        "cross-entropy against Dice against the combination, implement mean IoU and Dice metrics, and show "
        "the effect of removing the skip connections.",
    dataset="shapes_seg.npz",
    tasks=[
        ("core", "U-Net implementation",
         "Implement DoubleConv, Down, Up and OutConv blocks, and assemble a U-Net with depth 3. Verify that "
         "input and output spatial dimensions match for a 96x96 input."),
        ("core", "Train and visualise",
         "Train for 15 epochs and produce a figure with image, ground-truth mask and prediction for six "
         "validation scenes."),
        ("core", "mIoU and Dice from scratch",
         "Implement both metrics with a confusion-matrix accumulator. Report per-class IoU and the mean, and "
         "verify the background class is handled as you intend."),
        ("core", "Loss comparison",
         "Train three models with cross-entropy, Dice, and CE + Dice. Report mIoU for each and state which "
         "classes each loss favours."),
        ("core", "Skip connections ablation",
         "Remove the skip connections and retrain. Report the mIoU drop and show a side-by-side prediction "
         "demonstrating the loss of boundary detail."),
        ("core", "Class imbalance",
         "Report the pixel frequency of each class. Apply inverse-frequency class weights to cross-entropy and "
         "report the change in per-class IoU for the rarest class."),
        ("stretch", "Checkerboard artefacts",
         "Replace bilinear upsampling with ConvTranspose2d(k=3, stride=2) and show the checkerboard artefacts "
         "in the output. Then fix them with k=4 and explain why."),
        ("stretch", "Boundary-aware evaluation",
         "Implement boundary IoU (IoU restricted to a band around the object boundary). Report it alongside mIoU "
         "and explain what it reveals that mIoU does not."),
    ],
    reading=[
        "Ronneberger et al. (2015), *U-Net: Convolutional Networks for Biomedical Image Segmentation*.",
        "Long et al. (2015), *Fully Convolutional Networks for Semantic Segmentation*.",
        "Chen et al. (2018), *DeepLabv3+*.",
        "Odena et al. (2016), *Deconvolution and Checkerboard Artifacts*.",
    ],
)

lecture(
    number=12, part=3,
    title="Sequence Models: RNNs, LSTMs and Image Captioning",
    tagline="Memory, and the reason it eventually was not enough.",
    objectives=[
        "Explain recurrence, parameter sharing across time, and backpropagation through time.",
        "Derive why vanilla RNN gradients vanish, and how gating in an LSTM or GRU addresses it.",
        "Build a character-level language model and a CNN-encoder / RNN-decoder captioning model.",
        "Articulate the sequential bottleneck that motivated the Transformer.",
    ],
    outline=[
        ("Recurrence", [
            "h_t = tanh(W_hh h_{t-1} + W_xh x_t + b) - the same weights at every timestep.",
            "Weight sharing across time is the sequence analogue of weight sharing across space in a CNN.",
            "Configurations: one-to-many (captioning), many-to-one (classification), many-to-many (translation).",
        ]),
        ("Backpropagation through time", [
            "Unroll the network over the sequence and apply standard backpropagation.",
            "The gradient through T steps contains W_hh raised to the power T.",
            "Spectral radius below 1 vanishes; above 1 explodes. There is no comfortable middle for long sequences.",
            "Truncated BPTT bounds the cost; gradient clipping handles the explosion.",
        ]),
        ("LSTM", [
            "A cell state runs through the sequence with only additive interactions - an uninterrupted gradient path.",
            "Forget gate decides what to discard, input gate what to write, output gate what to expose.",
            "Initialise the forget-gate bias to 1: remember by default, learn to forget.",
            "GRU merges cell and hidden state into two gates - fewer parameters, usually comparable accuracy.",
        ]),
        ("Text as input", [
            "Tokenisation: character, word, or subword (BPE / WordPiece). Subword is the modern default.",
            "Embeddings map discrete token ids to dense learned vectors.",
            "Padding plus a mask, or pack_padded_sequence, for variable-length batches.",
            "Never let the loss count padded positions.",
        ]),
        ("Image captioning", [
            "CNN encoder produces a feature vector; RNN decoder generates tokens conditioned on it.",
            "Teacher forcing: feed the ground-truth token during training, the model's own output at inference.",
            "That mismatch is exposure bias; scheduled sampling is one partial remedy.",
            "Decoding: greedy is fast, beam search is better, sampling with temperature is more diverse.",
        ]),
        ("The bottleneck", [
            "Recurrence is inherently sequential - timestep t cannot start before t-1 finishes. No parallelism.",
            "A single fixed-size hidden state must summarise the entire history.",
            "Attention was invented to relieve exactly this, and then replaced recurrence altogether.",
        ]),
    ],
    lab="Implement an RNN cell and an LSTM cell from scratch and verify against PyTorch; train a character-level "
        "language model on the corpus and sample from it; demonstrate the vanishing-gradient difference between "
        "RNN and LSTM on a long-range copy task; then build CNN-encoder / LSTM-decoder captioning on the shapes "
        "images with greedy and beam-search decoding.",
    dataset="corpus.txt, sentiment.csv, captions.csv, shapes_32.npz",
    tasks=[
        ("core", "RNN cell from scratch",
         "Implement a vanilla RNN cell with explicit weight matrices. Verify one timestep against "
         "torch.nn.RNNCell with copied weights."),
        ("core", "LSTM cell from scratch",
         "Implement all four gates explicitly. Verify against torch.nn.LSTMCell with copied weights, and print "
         "the gate activations for one input to show what each gate is doing."),
        ("core", "Character language model",
         "Train a character-level LSTM on corpus.txt. Report perplexity and sample 300 characters at "
         "temperatures 0.5, 1.0 and 1.5. Comment on the difference."),
        ("core", "Long-range copy task",
         "Build a task requiring the model to reproduce a token seen T steps earlier. Plot accuracy against T "
         "for a vanilla RNN and an LSTM, for T in {5, 10, 25, 50, 100}."),
        ("core", "Gradient flow comparison",
         "Measure the gradient norm at timestep 0 for both architectures on a 100-step sequence. Report the ratio "
         "and connect it to the copy-task result."),
        ("core", "Image captioning",
         "Train a CNN encoder with an LSTM decoder on the shapes captions. Generate captions for eight test "
         "images with greedy decoding and report how many are factually correct."),
        ("stretch", "Beam search",
         "Implement beam search with width 3 and length normalisation. Compare its captions against greedy "
         "decoding on the same images."),
        ("stretch", "Gradient clipping study",
         "Train the vanilla RNN without clipping until the loss becomes NaN. Add clipping at norms 0.5, 1 and 5 "
         "and report which values keep training stable."),
    ],
    reading=[
        "Hochreiter & Schmidhuber (1997), *Long Short-Term Memory*.",
        "Karpathy (2015), *The Unreasonable Effectiveness of Recurrent Neural Networks*.",
        "Vinyals et al. (2015), *Show and Tell: A Neural Image Caption Generator*.",
        "Pascanu et al. (2013), *On the Difficulty of Training Recurrent Neural Networks*.",
    ],
)

lecture(
    number=13, part=4,
    title="Attention and the Transformer",
    tagline="Scaled dot-product attention, built from nothing.",
    objectives=[
        "Derive scaled dot-product attention and explain every term, including the square-root scaling.",
        "Implement multi-head self-attention, positional encoding and a full encoder block from scratch.",
        "Distinguish encoder-only, decoder-only and encoder-decoder architectures and their uses.",
        "Analyse attention's quadratic cost and name the main mitigations.",
    ],
    outline=[
        ("The idea", [
            "Every position attends to every other position directly. Path length between any two tokens is 1.",
            "Query, Key, Value: the query asks, keys advertise, values carry the content.",
            "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V.",
            "The sqrt(d_k) matters: without it, dot products grow with dimension and softmax saturates to one-hot.",
        ]),
        ("Multi-head attention", [
            "Project into h subspaces, attend independently in each, concatenate, project out.",
            "Different heads specialise - syntax, coreference, positional patterns - without being told to.",
            "Cost is unchanged: each head uses d_model / h dimensions.",
        ]),
        ("Position", [
            "Attention is permutation-equivariant - it has no notion of order at all.",
            "Sinusoidal encodings: fixed, and extrapolate to unseen lengths.",
            "Learned absolute embeddings: simple, standard in BERT and ViT.",
            "Rotary (RoPE) and ALiBi: relative schemes that dominate modern large language models.",
        ]),
        ("The Transformer block", [
            "Sub-layer 1: multi-head self-attention. Sub-layer 2: position-wise feed-forward (usually 4x width).",
            "Residual connection plus LayerNorm around each sub-layer.",
            "Pre-norm (norm before the sub-layer) trains far more stably than the original post-norm.",
            "Causal masking in the decoder prevents a position from attending to the future.",
        ]),
        ("Three families", [
            "Encoder-only (BERT): bidirectional context, for classification and retrieval.",
            "Decoder-only (GPT): causal, for generation. The dominant modern architecture.",
            "Encoder-decoder (T5, original Transformer): for sequence-to-sequence tasks like translation.",
        ]),
        ("The cost", [
            "Attention is O(n^2) in sequence length, in both time and memory.",
            "FlashAttention removes the memory cost with tiling and recomputation - exact, not approximate.",
            "Sparse and linear attention approximate the matrix; sliding windows bound the span.",
            "In practice the feed-forward layers hold most of the parameters; attention holds most of the cost at long n.",
        ]),
    ],
    lab="Implement scaled dot-product attention, multi-head attention, sinusoidal positional encoding and a "
        "full pre-norm encoder block from scratch, verifying each against torch.nn.MultiheadAttention; train "
        "a Transformer classifier on the sentiment corpus; visualise attention weights; and measure the "
        "quadratic scaling empirically.",
    dataset="sentiment.csv, corpus.txt",
    tasks=[
        ("core", "Scaled dot-product attention",
         "Implement attention(Q, K, V, mask=None) with correct masking. Verify against "
         "F.scaled_dot_product_attention to within 1e-5."),
        ("core", "Why sqrt(d_k)",
         "For d_k in {8, 64, 512}, sample random Q and K, and plot the distribution of the attention weights "
         "with and without the scaling. Show numerically that softmax saturates without it."),
        ("core", "Multi-head attention",
         "Implement MultiHeadAttention as a module. Verify against torch.nn.MultiheadAttention with copied "
         "weights, being careful about its packed in_proj_weight layout."),
        ("core", "Positional encoding",
         "Implement sinusoidal encoding and plot the resulting matrix as a heatmap. Then show that removing it "
         "entirely leaves a sentence-order task at chance accuracy."),
        ("core", "Encoder block and classifier",
         "Assemble a pre-norm encoder block and build a two-layer Transformer classifier. Train on sentiment.csv "
         "to at least 95% test accuracy."),
        ("core", "Attention visualisation",
         "Plot the attention matrix for each head on two example sentences. Describe in three sentences any "
         "pattern a head appears to have specialised in."),
        ("core", "Causal masking",
         "Implement a causal mask and prove by an ablation experiment that removing it lets a language model "
         "cheat - report the implausibly low loss it achieves."),
        ("stretch", "Quadratic scaling",
         "Measure forward-pass time and peak memory for sequence lengths 64, 128, 256, 512 and 1024. Fit the "
         "exponent of the scaling curve and compare it to the theoretical 2."),
        ("stretch", "Pre-norm versus post-norm",
         "Train a 6-layer Transformer in both configurations without warmup. Report which one diverges and "
         "explain why in terms of residual-stream magnitude."),
    ],
    reading=[
        "Vaswani et al. (2017), *Attention Is All You Need*.",
        "Alammar, *The Illustrated Transformer*.",
        "Xiong et al. (2020), *On Layer Normalization in the Transformer Architecture*.",
        "Dao et al. (2022), *FlashAttention*.",
    ],
)

lecture(
    number=14, part=4,
    title="Vision Transformers and Self-Supervised Learning",
    tagline="Images as sequences of patches, and learning without labels.",
    objectives=[
        "Explain the ViT patch-embedding pipeline and where its inductive bias differs from a CNN's.",
        "Implement a Vision Transformer from scratch and train it on the shapes dataset.",
        "Describe contrastive (SimCLR, MoCo), distillation (DINO) and masked (MAE) self-supervision.",
        "Run a linear probe to quantify the quality of a learned representation.",
    ],
    objectives_note="",
    outline=[
        ("Vision Transformer", [
            "Split the image into fixed patches (16x16 in the original), flatten, and linearly project each.",
            "Prepend a learnable [CLS] token; add positional embeddings; feed a standard Transformer encoder.",
            "Classify from the [CLS] output, or from mean-pooled patch tokens.",
        ]),
        ("Inductive bias and the data requirement", [
            "A CNN has locality and translation equivariance baked in. ViT has almost none.",
            "ViT underperforms ResNets on ImageNet-1k alone; it wins once pre-trained on JFT-300M.",
            "'Attention is all you need - given enough data.' With little data, the CNN prior is a real advantage.",
            "Hybrids (ConvNeXt, Swin) reintroduce locality and hierarchy and get the best of both.",
        ]),
        ("Why self-supervision", [
            "Labels are expensive; unlabelled images are effectively free.",
            "A pretext task manufactures supervision from the data's own structure.",
            "Early pretext tasks: rotation prediction, jigsaw, colourisation. Superseded, but instructive.",
        ]),
        ("Contrastive learning", [
            "SimCLR: two augmented views of one image are positives; every other image in the batch is a negative.",
            "NT-Xent loss with a temperature; the projection head is discarded after pre-training.",
            "It needs large batches for enough negatives. MoCo solves this with a momentum-updated queue.",
            "Augmentation choice is not a detail - it *is* the definition of what the model treats as invariant.",
        ]),
        ("Beyond contrastive", [
            "BYOL and DINO work without negatives, using a momentum teacher and stop-gradient to avoid collapse.",
            "MAE: mask 75% of patches and reconstruct them. Simple, highly scalable, very effective for ViT.",
            "The high mask ratio is what makes the task hard enough to force semantic understanding.",
        ]),
        ("Evaluating a representation", [
            "Linear probe: freeze the encoder, train a linear classifier. The standard measure of representation quality.",
            "k-NN probe: no training at all, just nearest neighbours in feature space.",
            "Few-shot transfer: the metric that matters when labels are genuinely scarce.",
        ]),
    ],
    lab="Implement patch embedding, the [CLS] token and a full ViT from scratch, train it on shapes and compare "
        "against the Lecture 6 CNN at matched parameter counts; then run SimCLR pre-training on the unlabelled "
        "pool and measure linear-probe accuracy against a supervised model trained on the same small label budget.",
    dataset="shapes_32.npz, shapes_pairs.npz",
    tasks=[
        ("core", "Patch embedding",
         "Implement PatchEmbed with a strided convolution. For a 32x32 input with patch size 4, verify the output "
         "is (B, 64, embed_dim) and explain the 64."),
        ("core", "Vision Transformer",
         "Assemble a ViT with the [CLS] token, learned positional embeddings and the encoder blocks from "
         "Lecture 13. Train it on shapes and report accuracy and parameter count."),
        ("core", "ViT versus CNN under data scarcity",
         "Train both at matched parameter counts on 500, 2000 and 6000 training examples. Plot accuracy against "
         "training-set size and explain the crossover in terms of inductive bias."),
        ("core", "Attention distance",
         "For each ViT layer, compute the mean attention distance in pixels. Plot it against depth and compare "
         "the early layers to a CNN's receptive field."),
        ("core", "SimCLR augmentations and NT-Xent",
         "Implement the two-view augmentation pipeline and the NT-Xent loss with temperature 0.5. Verify the loss "
         "on a hand-constructed batch where you know the right answer."),
        ("core", "Pre-train and probe",
         "Pre-train the encoder on shapes_pairs.npz for 20 epochs using no labels at all. Then train a linear "
         "probe on 100 labelled examples. Compare against supervised training on the same 100 examples."),
        ("stretch", "Augmentation ablation for SimCLR",
         "Remove colour jitter, then remove random cropping. Report linear-probe accuracy for each and explain "
         "why cropping matters most."),
        ("stretch", "Masked autoencoder",
         "Implement a minimal MAE: mask 75% of patches, reconstruct with a light decoder. Visualise "
         "reconstructions and report linear-probe accuracy against your SimCLR result."),
    ],
    reading=[
        "Dosovitskiy et al. (2021), *An Image Is Worth 16x16 Words* (ViT).",
        "Chen et al. (2020), *A Simple Framework for Contrastive Learning* (SimCLR).",
        "He et al. (2022), *Masked Autoencoders Are Scalable Vision Learners*.",
        "Caron et al. (2021), *Emerging Properties in Self-Supervised Vision Transformers* (DINO).",
    ],
)

lecture(
    number=15, part=4,
    title="Generative Models: Autoencoders, VAEs, GANs and Diffusion",
    tagline="Learning the distribution, not just the decision boundary.",
    objectives=[
        "Contrast discriminative and generative modelling and the trade-offs between the major families.",
        "Derive the VAE evidence lower bound and implement the reparameterisation trick.",
        "Train a GAN, recognise mode collapse, and apply the standard stabilisation techniques.",
        "Explain the forward and reverse diffusion processes and implement a minimal DDPM.",
    ],
    outline=[
        ("The generative families", [
            "Autoregressive (PixelCNN, GPT): exact likelihood, slow sequential sampling.",
            "VAE: latent variable model, fast sampling, blurry samples.",
            "GAN: sharp samples, unstable training, no likelihood.",
            "Diffusion: high quality and stable training, at the cost of many sampling steps.",
            "Normalising flows: exact likelihood and invertibility, with heavy architectural constraints.",
        ]),
        ("Autoencoders", [
            "Encoder compresses to a bottleneck; decoder reconstructs. Trained on reconstruction error.",
            "A plain autoencoder is not generative - the latent space has no structure to sample from.",
            "Useful for compression, denoising and anomaly detection via reconstruction error.",
        ]),
        ("Variational autoencoders", [
            "Encode to a distribution - a mean and a variance - rather than a point.",
            "ELBO = reconstruction term - KL(q(z|x) || p(z)). Maximise it to bound the log-likelihood.",
            "Reparameterisation: z = mu + sigma * eps, eps ~ N(0, I). This is what makes it differentiable.",
            "Posterior collapse: the KL term wins, the latent is ignored. KL annealing and beta-VAE address it.",
            "Samples are blurry because the pixel-wise Gaussian likelihood averages over plausible outputs.",
        ]),
        ("Generative adversarial networks", [
            "Generator versus discriminator, a minimax game with a Nash equilibrium as the goal.",
            "Non-saturating generator loss, because the original minimax form gives no gradient early on.",
            "Mode collapse: the generator finds a handful of outputs that fool the discriminator and stops there.",
            "Stabilisers: spectral normalisation, WGAN-GP, two time-scale update rules, label smoothing.",
            "Evaluation: FID is the standard, but it is sensitive to sample count and preprocessing.",
        ]),
        ("Diffusion models", [
            "Forward process: add Gaussian noise over T steps until the image is pure noise. Fixed, no learning.",
            "Reverse process: a network learns to predict the noise added at each step.",
            "Training is a simple regression on the noise - remarkably stable compared to a GAN.",
            "Sampling is iterative and slow; DDIM and distillation cut the step count substantially.",
            "Classifier-free guidance trades diversity for fidelity and is what makes text-to-image work.",
        ]),
        ("Choosing, and the ethics", [
            "Need likelihoods? Autoregressive or flows. Need speed? VAE or GAN. Need quality? Diffusion.",
            "Deepfakes, consent and provenance are engineering concerns, not only policy ones.",
            "Training data attribution and memorisation are unresolved. Know what your model was trained on.",
        ]),
    ],
    lab="Train an autoencoder and visualise its latent space; implement a VAE with the reparameterisation trick "
        "and interpolate between latent codes; train a DCGAN on shapes and deliberately induce and then fix mode "
        "collapse; implement a minimal DDPM with a small U-Net and visualise the reverse process step by step.",
    dataset="shapes_32.npz, digits_8x8.npz",
    tasks=[
        ("core", "Autoencoder and latent space",
         "Train an autoencoder with a 2-dimensional bottleneck on shapes. Scatter-plot the latent codes coloured "
         "by class and state whether the classes separate."),
        ("core", "VAE with reparameterisation",
         "Implement the encoder producing mu and logvar, the reparameterisation trick, and the ELBO with a "
         "closed-form Gaussian KL. Plot reconstruction loss and KL separately across training."),
        ("core", "Sample and interpolate",
         "Sample 64 images from the VAE prior. Then linearly interpolate between two latent codes in 10 steps and "
         "show the decoded sequence."),
        ("core", "beta-VAE",
         "Train with beta in {0.5, 1, 4, 10}. Show the reconstruction/KL trade-off in a table and identify where "
         "posterior collapse begins."),
        ("core", "DCGAN",
         "Implement a DCGAN generator and discriminator. Train on shapes and produce a sample grid every five "
         "epochs to show the progression."),
        ("core", "Mode collapse",
         "Induce mode collapse by over-training the generator relative to the discriminator. Demonstrate it "
         "quantitatively with a class histogram over 1000 samples, then fix it and show the histogram recover."),
        ("core", "Minimal DDPM",
         "Implement the cosine noise schedule, the forward diffusion q(x_t | x_0) in closed form, and a small "
         "noise-prediction U-Net. Train it and visualise the reverse process at t = T, 3T/4, T/2, T/4, 0."),
        ("stretch", "DDIM sampling",
         "Implement deterministic DDIM sampling. Compare sample quality and wall-clock time at 1000, 100, 50 and "
         "10 steps."),
        ("stretch", "Conditional generation",
         "Add class conditioning to your diffusion model via an embedding added to the time embedding. Generate "
         "each class on demand and verify with your Lecture 6 classifier."),
    ],
    reading=[
        "Kingma & Welling (2014), *Auto-Encoding Variational Bayes*.",
        "Goodfellow et al. (2014), *Generative Adversarial Networks*.",
        "Ho et al. (2020), *Denoising Diffusion Probabilistic Models*.",
        "Radford et al. (2016), *DCGAN*.",
        "Ho & Salimans (2022), *Classifier-Free Diffusion Guidance*.",
    ],
)

lecture(
    number=16, part=4,
    title="Multimodal Learning and Production ML Systems",
    tagline="Joining vision to language, and shipping the result responsibly.",
    objectives=[
        "Explain contrastive vision-language pre-training and how it enables zero-shot classification.",
        "Implement a small CLIP-style dual encoder and perform zero-shot and retrieval tasks.",
        "Design a production ML system: versioning, monitoring, drift detection and rollback.",
        "Audit a model for robustness, fairness and calibration before release.",
    ],
    outline=[
        ("Multimodal learning", [
            "Fusion strategies: early (concatenate inputs), late (combine predictions), joint embedding.",
            "CLIP: an image encoder and a text encoder trained to agree on matched pairs, over 400M of them.",
            "Symmetric InfoNCE over the image-text similarity matrix, with a learned temperature.",
            "Zero-shot classification: embed the class names as text prompts and take the nearest.",
        ]),
        ("What multimodal buys you", [
            "Zero-shot transfer to classes never explicitly labelled during training.",
            "Text-to-image and image-to-text retrieval from the same shared embedding space.",
            "The backbone for text-to-image generation and for vision-language models.",
            "Limits: prompt sensitivity, weak compositional reasoning, and inherited web-scale bias.",
        ]),
        ("Production architecture", [
            "Offline training pipeline versus the online serving path - different constraints, different code.",
            "A feature or preprocessing store keeps training and serving transformations identical.",
            "Model registry: versioned artefacts with metrics, lineage and an approval stage.",
            "Shadow deployment, then canary, then full rollout. Keep the rollback path one command away.",
        ]),
        ("Monitoring", [
            "Operational: latency percentiles, throughput, error rate, saturation.",
            "Input drift: distribution shift in the incoming data, detectable without any labels.",
            "Prediction drift: a shift in the output distribution - the earliest available warning.",
            "Performance: needs labels, which usually arrive late or never. Plan the feedback loop deliberately.",
            "Set alert thresholds from a real baseline, not from intuition.",
        ]),
        ("Pre-release audit", [
            "Robustness: corruption benchmarks, adversarial examples, out-of-distribution detection.",
            "Fairness: subgroup performance, not aggregate accuracy. Pick a fairness definition and justify it.",
            "Calibration: a confidence score that downstream systems will act on must actually mean something.",
            "Documentation: a model card stating intended use, training data, evaluation and known limitations.",
        ]),
        ("Where the field is going", [
            "Scaling laws, and the growing dominance of compute-optimal training.",
            "Foundation models plus lightweight adaptation (LoRA, adapters, prompt tuning).",
            "Efficiency: quantisation, distillation, pruning - the difference between a demo and a product.",
            "Open problems: reasoning, sample efficiency, reliable uncertainty, and verifiable alignment.",
        ]),
    ],
    lab="Build a CLIP-style dual encoder over the shapes images and their captions, train it with symmetric "
        "InfoNCE, run zero-shot classification from text prompts and bidirectional retrieval; then build the "
        "production layer: a versioned model registry, a FastAPI service with monitoring and drift detection "
        "endpoints, and a complete pre-release audit with a model card.",
    dataset="shapes_32.npz, captions.csv, shapes_pairs.npz",
    app="Production FastAPI service in app/ - registry, monitoring, drift detection, model card",
    tasks=[
        ("core", "Dual encoder",
         "Implement an image encoder (small CNN) and a text encoder (embedding + mean pooling or a small "
         "Transformer) projecting into a shared 64-dimensional L2-normalised space."),
        ("core", "Symmetric contrastive loss",
         "Implement InfoNCE in both directions with a learnable temperature. Verify that a perfectly aligned "
         "batch gives near-zero loss."),
        ("core", "Zero-shot classification",
         "Classify test images using only the text prompts 'a photo of a {class}'. Report accuracy and compare "
         "against the supervised Lecture 6 model."),
        ("core", "Prompt sensitivity",
         "Try five different prompt templates. Report the accuracy spread and comment on what that implies for "
         "deploying a zero-shot system."),
        ("core", "Bidirectional retrieval",
         "Implement text-to-image and image-to-text retrieval. Report Recall@1 and Recall@5 for both directions."),
        ("core", "Model registry and serving",
         "Build a registry that versions checkpoints with their metrics and config. Serve the latest approved "
         "version from FastAPI with GET /models and POST /models/{version}/promote."),
        ("core", "Drift detection",
         "Implement a /metrics endpoint tracking the prediction distribution and a population stability index "
         "against the training baseline. Feed it corrupted images and show the PSI alarm fire."),
        ("core", "Model card",
         "Write a complete model card: intended use, out-of-scope use, training data, evaluation results, "
         "subgroup performance, calibration, known limitations and the rollback procedure."),
        ("stretch", "Robustness benchmark",
         "Evaluate under five corruptions (Gaussian noise, blur, brightness, contrast, rotation) at three "
         "severities. Produce a 5x3 accuracy table and identify the weakest axis."),
        ("stretch", "Quantisation",
         "Apply dynamic quantisation to the served model. Report model size, p95 latency and accuracy before "
         "and after, and state whether you would ship it."),
    ],
    reading=[
        "Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision* (CLIP).",
        "Mitchell et al. (2019), *Model Cards for Model Reporting*.",
        "Sculley et al. (2015), *Hidden Technical Debt in Machine Learning Systems*.",
        "Hendrycks & Dietterich (2019), *Benchmarking Neural Network Robustness to Common Corruptions*.",
    ],
)

assert len(LECTURES) == 16, f"expected 16 lectures, got {len(LECTURES)}"
for i, lec in enumerate(LECTURES, start=1):
    assert lec["number"] == i, f"lecture ordering broken at {i}"


def by_number(n: int) -> dict:
    return LECTURES[n - 1]


def folder_name(n: int) -> str:
    return f"Lecture_{n:02d}"


def slug(title: str) -> str:
    keep = [c if (c.isalnum() or c == " ") else "" for c in title]
    return "_".join("".join(keep).split())
