import numpy as np
import scipy.fft

class _MockFunction:
    def __init__(self, return_value = None):
        self.number_calls = 0
        self.return_value = return_value
        self.last_args = ([], {})

    def __call__(self, *args, **kwargs):
        self.number_calls += 1
        self.last_args = (args, kwargs)
        return self.return_value


fft = _MockFunction(np.random.random(10))
fft2 = _MockFunction(np.random.random(10))
fftn = _MockFunction(np.random.random(10))

ifft = _MockFunction(np.random.random(10))
ifft2 = _MockFunction(np.random.random(10))
ifftn = _MockFunction(np.random.random(10))

rfft = _MockFunction(np.random.random(10))
rfft2 = _MockFunction(np.random.random(10))
rfftn = _MockFunction(np.random.random(10))

irfft = _MockFunction(np.random.random(10))
irfft2 = _MockFunction(np.random.random(10))
irfftn = _MockFunction(np.random.random(10))

hfft = _MockFunction(np.random.random(10))
hfft2 = _MockFunction(np.random.random(10))
hfftn = _MockFunction(np.random.random(10))

ihfft = _MockFunction(np.random.random(10))
ihfft2 = _MockFunction(np.random.random(10))
ihfftn = _MockFunction(np.random.random(10))

dct = _MockFunction(np.random.random(10))
idct = _MockFunction(np.random.random(10))
dctn = _MockFunction(np.random.random(10))
idctn = _MockFunction(np.random.random(10))

dst = _MockFunction(np.random.random(10))
idst = _MockFunction(np.random.random(10))
dstn = _MockFunction(np.random.random(10))
idstn = _MockFunction(np.random.random(10))

fht = _MockFunction(np.random.random(10))
ifht = _MockFunction(np.random.random(10))


__ua_domain__ = "numpy.scipy.fft"


_implements = {
    scipy.fft.fft: fft,
    scipy.fft.fft2: fft2,
    scipy.fft.fftn: fftn,
    scipy.fft.ifft: ifft,
    scipy.fft.ifft2: ifft2,
    scipy.fft.ifftn: ifftn,
    scipy.fft.rfft: rfft,
    scipy.fft.rfft2: rfft2,
    scipy.fft.rfftn: rfftn,
    scipy.fft.irfft: irfft,
    scipy.fft.irfft2: irfft2,
    scipy.fft.irfftn: irfftn,
    scipy.fft.hfft: hfft,
    scipy.fft.hfft2: hfft2,
    scipy.fft.hfftn: hfftn,
    scipy.fft.ihfft: ihfft,
    scipy.fft.ihfft2: ihfft2,
    scipy.fft.ihfftn: ihfftn,
    scipy.fft.dct: dct,
    scipy.fft.idct: idct,
    scipy.fft.dctn: dctn,
    scipy.fft.idctn: idctn,
    scipy.fft.dst: dst,
    scipy.fft.idst: idst,
    scipy.fft.dstn: dstn,
    scipy.fft.idstn: idstn,
    scipy.fft.fht: fht,
    scipy.fft.ifht: ifht
}


def __ua_function__(method, args, kwargs):
    fn = _implements.get(method)
    return (fn(*args, **kwargs) if fn is not None
            else NotImplemented)
