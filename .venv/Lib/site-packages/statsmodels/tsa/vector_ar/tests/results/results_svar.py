"""
Test Results for the SVAR model. Obtained from R using svartest.R
"""


class SVARdataResults:
    def __init__(self):
        self.A = ([
            [1.0, 0.0, 0],
            [-0.506802245, 1.0, 0],
            [-5.536056520, 3.04117686, 1.0]])
        self.B = ([
            [0.0075756676, 0.0, 0.0],
            [0.0, 0.00512051886, 0.0],
            [0.0, 0.0, 0.020708948]])
