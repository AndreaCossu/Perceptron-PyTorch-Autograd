# PyTorch implementation of Single Layer Perceptron to separate linearly separable data.

A single pattern of data is a 2-dimensional point in the cartesian plane with (-1, 1) labels.

The list of point is stored in data.csv file.

The perceptron implementation can use 3 different gradient computation method:
* Backward - it uses PyTorch loss.backward() method to compute derivative of loss with respect to all of its leaves
* Grad - it uses PyTorch torch.autograd.grad(loss, x) to compute derivative of loss with respect to x.
* Optimizer - it uses the default SGD optimizer of PyTorch.
