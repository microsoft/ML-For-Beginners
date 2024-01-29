# Fused types for input like y_true, raw_prediction, sample_weights.
ctypedef fused floating_in:
    double
    float


# Fused types for output like gradient and hessian
# We use a different fused types for input (floating_in) and output (floating_out), such
# that input and output can have different dtypes in the same function call. A single
# fused type can only take on one single value (type) for all arguments in one function
# call.
ctypedef fused floating_out:
    double
    float


# Struct to return 2 doubles
ctypedef struct double_pair:
    double val1
    double val2


# C base class for loss functions
cdef class CyLossFunction:
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil


cdef class CyHalfSquaredError(CyLossFunction):
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil


cdef class CyAbsoluteError(CyLossFunction):
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil


cdef class CyPinballLoss(CyLossFunction):
    cdef readonly double quantile  # readonly makes it accessible from Python
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil


cdef class CyHuberLoss(CyLossFunction):
    cdef public double delta  # public makes it accessible from Python
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil


cdef class CyHalfPoissonLoss(CyLossFunction):
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil


cdef class CyHalfGammaLoss(CyLossFunction):
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil


cdef class CyHalfTweedieLoss(CyLossFunction):
    cdef readonly double power  # readonly makes it accessible from Python
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil


cdef class CyHalfTweedieLossIdentity(CyLossFunction):
    cdef readonly double power  # readonly makes it accessible from Python
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil


cdef class CyHalfBinomialLoss(CyLossFunction):
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil


cdef class CyExponentialLoss(CyLossFunction):
    cdef double cy_loss(self, double y_true, double raw_prediction) noexcept nogil
    cdef double cy_gradient(self, double y_true, double raw_prediction) noexcept nogil
    cdef double_pair cy_grad_hess(self, double y_true, double raw_prediction) noexcept nogil
