# data/

## mnist/

The four standard MNIST files, vendored into the repository so the container
needs no network access at run time (the original download host has already
moved once, which would silently break a clean-clone reproduction).

    train-images-idx3-ubyte.gz   train-labels-idx1-ubyte.gz
    t10k-images-idx3-ubyte.gz    t10k-labels-idx1-ubyte.gz

**Source and credit.** MNIST is the work of Yann LeCun, Corinna Cortes and
Christopher J. C. Burges, derived from NIST Special Database 19. It is
distributed for research use and is redistributed here unmodified, purely as
a convenience for reproducibility.

> Y. LeCun, L. Bottou, Y. Bengio, P. Haffner. *Gradient-based learning applied
> to document recognition.* Proceedings of the IEEE, 86(11):2278-2324, 1998.

The files are byte-for-byte the standard release; we did not alter them. The
Dockerfile copies them to `/tmp/nngp/data/`, which is where the original
`load_dataset.py` looks by default.

CIFAR-10 is **not** vendored. `uncertainty_plot.py --dataset=cifar10`
downloads it at run time (Krizhevsky, 2009), so it is not part of the default
clean-clone run.
