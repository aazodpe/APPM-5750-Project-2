# APPM 5750 Project 2 -- NNGP Figure 3 reproduction + GELU extension
#
# Clean-clone test (what the grader runs):
#   git clone <repo-url> && cd <repo-name>
#   docker build -t nngp-project .
#   docker run nngp-project
#
# To get the PNG/CSV outputs onto your machine, mount a folder:
#   docker run -v "$(pwd)/output":/nngp/output nngp-project          (macOS/Linux)
#   docker run -v "${PWD}/output:/nngp/output" nngp-project          (PowerShell)
#
# Other modes (first argument to `docker run nngp-project ...`):
#   fig3 | extension | validate | all (default)
#   or any --flags, which are passed straight to uncertainty_plot.py

# ---------------------------------------------------------------------------
# Base image (from class): TF 1.15 on Python 3.6. Only an amd64 build exists,
# so pin the platform here -- then a plain `docker build` also works on
# Apple-silicon Macs (runs under emulation) without --platform flags.
# ---------------------------------------------------------------------------
FROM --platform=linux/amd64 tensorflow/tensorflow:1.15.0-py3

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /nngp

# Dependencies first (separate layer => cached across code edits).
# matplotlib is not in the base image. Everything is pinned to the last
# releases that support Python 3.6 so the build can't drift over time.
RUN pip install --no-cache-dir \
        matplotlib==3.3.4 \
        pillow==8.4.0 \
        kiwisolver==1.3.1 \
        cycler==0.11.0 \
        pyparsing==2.4.7 \
        python-dateutil==2.8.2

# Original nngp source + grid_data/ + our scripts + vendored MNIST.
COPY . /nngp

# --- Base-environment fixes from class -------------------------------------
#  1. Python 2 -> 3: xrange no longer exists.
#  2. numpy >= 1.16.3 defaults to allow_pickle=False, but grid_data/ files are
#     pickled object arrays written under Python 2 (hence latin1).
# --- Our additions ---------------------------------------------------------
#  3. Strip Windows CRLF from the shell script in case the repo was cloned on
#     Windows with core.autocrlf=true (would break bash with "\r" errors).
#  4. Put the vendored MNIST .gz files where load_dataset.py looks by
#     default, so nothing is downloaded at run time (the original download
#     host for MNIST has already moved once).
#  5. Build the GELU kernel lookup grid only if it is missing (it is shipped
#     in grid_data/, regenerating takes ~10 min).
RUN sed -i 's/\bxrange\b/range/g' *.py && \
    sed -i "s/np.load(f)/np.load(f, allow_pickle=True, encoding='latin1')/" nngp.py && \
    sed -i 's/\r$//' run_all.sh && \
    mkdir -p /tmp/nngp/data && cp data/mnist/*.gz /tmp/nngp/data/ && \
    (test -f grid_data/grid_gelu_ng501_ns501_nc500_mv100_mg10 || \
        python make_grid.py --nonlinearity=gelu) && \
    python -c "import activations, gpr, nngp, uncertainty_plot, extension_gelu; print('imports OK')"

ENTRYPOINT ["bash", "run_all.sh"]
CMD ["all"]
