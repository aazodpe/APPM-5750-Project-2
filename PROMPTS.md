# LLM Pair-Programming Log

The course asked for pair programming on Part 3, with the LLM (Claude, Anthropic) as **driver** and me as **navigator**. This file records my prompts, what the LLM did, and what was checked. Sections marked **Navigator notes** are for my own review comments and follow-up prompts.

---

## Session 1: 2026-09-28

### Prompt 1

> Can you read the project instructions and get started with working on it? I've also provided the lecture notes discussing this project and the relevant files used in it. Also summarize what the project is and explain it to me.

Files provided: `project 2 instructions.pdf`, `Lecture 9 ... Project 2 Workday.pdf`, the in-class `Dockerfile` and `uncertainty_plot.py`.

**What the LLM did:** It read the assignment and lecture slides and summarized the paper and the rubric. It flagged problems in the starter files:

- `dockerignore` was missing its leading dot.
- A plain `docker run` leaves the PNG inside the container.
- The CIFAR loader didn't zero-mean the labels.

It recommended Option 2 and asked three multiple-choice questions.

### Prompt 2 (answers to the LLM's questions)

> Figure: **Option 2 (Fig 3 uncertainty)**. Extension: **New activation**. Setup: **None yet** (start from scratch).

**What the LLM did, step by step:**

1. **Environment.** It cloned `brain-research/nngp`. Docker Hub was blocked in its sandbox, so it rebuilt an equivalent base image: Ubuntu 18.04, Python 3.6, `pip install tensorflow==1.15.0`. It ran the *unchanged* in-class Dockerfile on that image, and the baseline MNIST Figure 3 worked (tanh 0.989 / ReLU 0.987 at 1k).
2. **Choice of activation.** It proposed **GELU** because GELU is widely used today, absent from the paper, and has no simple closed-form kernel, so it actually exercises the paper's Sec. 2.5 numerical grid.
3. **Problem found:** `nngp._compute_qmap_grid` allocates a `[501, 501, 500]` float64 tensor (about 1 GB) per variance row, one per CPU core, which exhausts a typical Docker VM. It also names grid files after `nonlin_fn.__name__`, so a `lambda` can't be used. The fix was `activations.py` (named `gelu`) and `make_grid.py` (the same quadrature in NumPy, chunked).
4. **Verification before trusting the new grid:**
   - It regenerated rows of the shipped tanh and ReLU grids: max difference 5e-12. The formula matches the original code.
   - It compared ReLU with the exact arc-cosine kernel. **Unexpected result:** 9% error at variance 20. Investigation showed that the original code's integration range is fixed at z ∈ [−10, 10], so Gaussian tails are cut off at large variance. The error is under 1.2e-4 for v ≤ 5, which is the regime used here. This is documented as a limitation instead of hidden.
   - It checked the GELU grid against Monte-Carlo estimates (within about 2.5 standard errors).
5. **Figure 3 at 5k training points, 10k test points:** correlations of 0.958 (tanh), 0.970 (ReLU) and 0.974 (GELU).
6. **Extension analysis:** a sweep over training sizes 250–5000, plus calibration metrics. It noticed the variance overestimates MSE by 5–11×, so it added the maximum-likelihood kernel amplitude (reusing `v = L⁻¹t` from `gpr.py`, so no extra solve). After rescaling, the slope is about 1.1–1.2 for all activations.
7. **Reproducibility hardening:** bundled MNIST (the download host was also unreachable from the sandbox), `.gitattributes` for LF line endings, `FROM --platform=linux/amd64`, pinned matplotlib dependencies, and a build-time import check.
8. **Clean-clone test:** `git clone` into an empty directory, then `docker build --no-cache`, then `docker run`.

**Navigator notes (to fill in):**
- [ ] Read `activations.py` and `make_grid.py`. Do the formulas match `_compute_qmap_grid` in `nngp.py`?
- [ ] Run `docker run nngp-project validate` myself.
- [ ] Check the claim in README §4.1 (3) against `results/ext_cmap.png`.
- [ ] Add the paper's Fig. 3 image and the paper's correlation numbers to README §2.
- [ ] Have a classmate run the clean-clone test (Lecture 9, "Test with a Friend").

---

## Pitfalls of AI-generated code seen in this project (for the interview)

- **Plausible but unverified numerics.** A new kernel grid "looks right" whatever it contains. Only the regression test against the shipped grids and the closed-form check made it trustworthy, and that check is what exposed the truncation limitation in the original code.
- **Silent assumptions in old code.** Grid files are named after `fn.__name__`, so a lambda would silently map to a file called `grid_<lambda>_...`. You only catch this by reading the original code, not the generated code.
- **"Works on my machine" in the LLM's sandbox.** The sandbox couldn't reach Docker Hub or the MNIST host. The final test still has to be a real clean clone on a normal machine.
- **Over-claiming.** A correlation of 0.97 was easy to describe as "well calibrated". The calibration slope showed it isn't until the amplitude is fitted.

## Follow-up prompts

*(Add any later prompts here, e.g. "explain eq. 9 in gpr.py line by line", "why does GELU's C-map sit below ReLU's?")*
