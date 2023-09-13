# cython: language_level=3

# Fused types for y_true, y_pred, raw_prediction
ctypedef fused Y_DTYPE_C:
    double
    float


# Fused types for gradient and hessian
ctypedef fused G_DTYPE_C:
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
