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
- [x] Add the paper's Fig. 3 image and the paper's correlation numbers to README §2.  *(done in session 2)*
- [ ] Have a classmate run the clean-clone test (Lecture 9, "Test with a Friend").

---

## Pitfalls of AI-generated code seen in this project (for the interview)

- **Plausible but unverified numerics.** A new kernel grid "looks right" whatever it contains. Only the regression test against the shipped grids and the closed-form check made it trustworthy, and that check is what exposed the truncation limitation in the original code.
- **Silent assumptions in old code.** Grid files are named after `fn.__name__`, so a lambda would silently map to a file called `grid_<lambda>_...`. You only catch this by reading the original code, not the generated code.
- **"Works on my machine" in the LLM's sandbox.** The sandbox couldn't reach Docker Hub or the MNIST host. The final test still has to be a real clean clone on a normal machine.
- **Over-claiming.** A correlation of 0.97 was easy to describe as "well calibrated". The calibration slope showed it isn't until the amplitude is fitted.

## Follow-up prompts

Later sessions are logged below. Questions still worth asking the LLM (or
answering myself) before the interview: "explain Eq. 9 in `gpr.py` line by
line", "why does GELU's C-map sit below ReLU's?"

---

## Session 2: 2026-09-29 -- audit and finishing pass

### Prompt 1

> Can you read the project instructions, and see if everything is made correctly. Let me know what you need from me for full completion.

**What the LLM did:** Re-read the assignment PDF and checked the repo against each rubric line. It then actually ran the grader's test rather than reasoning about it: cloned this repo into an empty temp directory, ran `docker build --no-cache`, and ran the container end to end. Both exited 0, and the container's numbers matched the committed `results/*.csv` exactly.

It reported five gaps: (1) `git remote` still pointed at `brain-research/nngp`, so `git clone <my-repo>` could not work; (2) `figures/paper_fig3.png` did not exist, leaving a broken image where the rubric asks for a side-by-side; (3) the paper's correlation numbers in README section 2 were placeholders; (4) section 4.1 was near the word limit; (5) stale in-class copies of `Dockerfile` and `uncertainty_plot.py` sit in the parent directory.

**Navigator note:** the useful part was that it ran the clean-clone test instead of eyeballing the Dockerfile. The build passing is the thing the rubric actually pays 25 points for.

### Prompt 2

> repo url: https://github.com/aazodpe/Stat-5720-Project-2.git -- I've added a paper which I think is the one we need.

*(The repo was later renamed to `APPM-5750-Project-2` to match the course
number on the Lecture 9 title slide; GitHub redirects the old URL.)*

**What the LLM did:**

1. Confirmed the PDF is the right paper (Lee et al., ICLR 2018) and found Figure 3 on page 8.
2. Extracted the legend values. The text layer interleaves the two panels, so it used word x-coordinates to decide which legend belongs to which panel: the MNIST-50k panel (x approx 135-300) carries Tanh 0.9330 / ReLU 0.9573, and CIFAR-45k (x approx 300-470) carries 0.7428 / 0.8223. This is worth knowing for the interview -- a naive text extraction pairs them the wrong way round.
3. Rendered page 8 at 300 dpi and cropped `figures/paper_fig3.png` (MNIST panel) and `figures/paper_fig3_both_panels.png`.
4. Filled in README section 2, corrected the runtime estimates (the measured run is about 12 minutes, not 5), and recorded the clean-clone test result in section 6.

**Navigator notes (to fill in):**
- [ ] Check the cropped figure against page 8 of the PDF myself.
- [ ] Confirm the 0.9330 / 0.9573 pairing by looking at the MNIST panel legend directly.

### On our numbers being *higher* than the paper's

Worth having an answer ready for the interview, since it looks backwards at first: we beat the paper's correlation (0.958 vs 0.933 tanh, 0.970 vs 0.957 ReLU) while using 10x less training data. The README's explanation is that the binned correlation is a statistic about the *spread* of the bins, not about accuracy. With 5k training points the posterior variances are larger and cover a wider range, so the 100-point bins spread out along the x-axis and the trend fits more cleanly. A higher binned correlation here does not mean a better model -- our accuracy is lower than the paper's. This is a good example of a metric that is not monotone in model quality.
