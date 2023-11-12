"""
Kalman Smoother

Author: Chad Fulton
License: Simplified-BSD
"""

import numpy as np

SMOOTHER_STATE = 0x01          # Durbin and Koopman (2012), Chapter 4.4.2
SMOOTHER_STATE_COV = 0x02      # ibid., Chapter 4.4.3
SMOOTHER_DISTURBANCE = 0x04    # ibid., Chapter 4.5
SMOOTHER_DISTURBANCE_COV = 0x08    # ibid., Chapter 4.5
SMOOTHER_ALL = (
    SMOOTHER_STATE | SMOOTHER_STATE_COV | SMOOTHER_DISTURBANCE |
    SMOOTHER_DISTURBANCE_COV
)


class _KalmanSmoother:

    def __init__(self, model, kfilter, smoother_output):
        # Save values
        self.model = model
        self.kfilter = kfilter
        self._kfilter = model._kalman_filter
        self.smoother_output = smoother_output

        # Create storage
        self.scaled_smoothed_estimator = None
        self.scaled_smoothed_estimator_cov = None
        self.smoothing_error = None
        self.smoothed_state = None
        self.smoothed_state_cov = None
        self.smoothed_state_disturbance = None
        self.smoothed_state_disturbance_cov = None
        self.smoothed_measurement_disturbance = None
        self.smoothed_measurement_disturbance_cov = None

        # Intermediate values
        self.tmp_L = np.zeros((model.k_states, model.k_states, model.nobs),
                              dtype=kfilter.dtype)

        if smoother_output & (SMOOTHER_STATE | SMOOTHER_DISTURBANCE):
            self.scaled_smoothed_estimator = (
                np.zeros((model.k_states, model.nobs+1), dtype=kfilter.dtype))
            self.smoothing_error = (
                np.zeros((model.k_endog, model.nobs), dtype=kfilter.dtype))
        if smoother_output & (SMOOTHER_STATE_COV | SMOOTHER_DISTURBANCE_COV):
            self.scaled_smoothed_estimator_cov = (
                np.zeros((model.k_states, model.k_states, model.nobs + 1),
                         dtype=kfilter.dtype))

        # State smoothing
        if smoother_output & SMOOTHER_STATE:
            self.smoothed_state = np.zeros((model.k_states, model.nobs),
                                           dtype=kfilter.dtype)
        if smoother_output & SMOOTHER_STATE_COV:
            self.smoothed_state_cov = (
                np.zeros((model.k_states, model.k_states, model.nobs),
                         dtype=kfilter.dtype))

        # Disturbance smoothing
        if smoother_output & SMOOTHER_DISTURBANCE:
            self.smoothed_state_disturbance = (
                np.zeros((model.k_posdef, model.nobs), dtype=kfilter.dtype))
            self.smoothed_measurement_disturbance = (
                np.zeros((model.k_endog, model.nobs), dtype=kfilter.dtype))
        if smoother_output & SMOOTHER_DISTURBANCE_COV:
            self.smoothed_state_disturbance_cov = (
                np.zeros((model.k_posdef, model.k_posdef, model.nobs),
                         dtype=kfilter.dtype))
            self.smoothed_measurement_disturbance_cov = (
                np.zeros((model.k_endog, model.k_endog, model.nobs),
                         dtype=kfilter.dtype))

    def seek(self, t):
        if t >= self.model.nobs:
            raise IndexError("Observation index out of range")
        self.t = t

    def __iter__(self):
        return self

    def __call__(self):
        self.seek(self.model.nobs-1)
        # Perform backwards smoothing iterations
        for i in range(self.model.nobs-1, -1, -1):
            next(self)

    def next(self):
        # next() is required for compatibility with Python2.7.
        return self.__next__()

    def __next__(self):
        # Check for valid iteration
        if not self.t >= 0:
            raise StopIteration

        # Get local copies of variables
        t = self.t
        kfilter = self.kfilter
        _kfilter = self._kfilter
        model = self.model
        smoother_output = self.smoother_output

        scaled_smoothed_estimator = self.scaled_smoothed_estimator
        scaled_smoothed_estimator_cov = self.scaled_smoothed_estimator_cov
        smoothing_error = self.smoothing_error
        smoothed_state = self.smoothed_state
        smoothed_state_cov = self.smoothed_state_cov
        smoothed_state_disturbance = self.smoothed_state_disturbance
        smoothed_state_disturbance_cov = self.smoothed_state_disturbance_cov
        smoothed_measurement_disturbance = (
            self.smoothed_measurement_disturbance)
        smoothed_measurement_disturbance_cov = (
            self.smoothed_measurement_disturbance_cov)
        tmp_L = self.tmp_L

        # Seek the Cython Kalman filter to the right place, setup matrices
        _kfilter.seek(t, False)
        _kfilter.initialize_statespace_object_pointers()
        _kfilter.initialize_filter_object_pointers()
        _kfilter.select_missing()

        missing_entire_obs = (
            _kfilter.model.nmissing[t] == _kfilter.model.k_endog)
        missing_partial_obs = (
            not missing_entire_obs and _kfilter.model.nmissing[t] > 0)

        # Get the appropriate (possibly time-varying) indices
        design_t = 0 if kfilter.design.shape[2] == 1 else t
        obs_cov_t = 0 if kfilter.obs_cov.shape[2] == 1 else t
        transition_t = 0 if kfilter.transition.shape[2] == 1 else t
        selection_t = 0 if kfilter.selection.shape[2] == 1 else t
        state_cov_t = 0 if kfilter.state_cov.shape[2] == 1 else t

        # Get endog dimension (can vary if there missing data)
        k_endog = _kfilter.k_endog

        # Get references to representation matrices and Kalman filter output
        transition = model.transition[:, :, transition_t]
        selection = model.selection[:, :, selection_t]
        state_cov = model.state_cov[:, :, state_cov_t]

        predicted_state = kfilter.predicted_state[:, t]
        predicted_state_cov = kfilter.predicted_state_cov[:, :, t]

        mask = ~kfilter.missing[:, t].astype(bool)
        if missing_partial_obs:
            design = np.array(
                _kfilter.selected_design[:k_endog*model.k_states], copy=True
            ).reshape(k_endog, model.k_states, order='F')
            obs_cov = np.array(
                _kfilter.selected_obs_cov[:k_endog**2], copy=True
            ).reshape(k_endog, k_endog)
            kalman_gain = kfilter.kalman_gain[:, mask, t]

            forecasts_error_cov = np.array(
                _kfilter.forecast_error_cov[:, :, t], copy=True
                ).ravel(order='F')[:k_endog**2].reshape(k_endog, k_endog)
            forecasts_error = np.array(
                _kfilter.forecast_error[:k_endog, t], copy=True)
            F_inv = np.linalg.inv(forecasts_error_cov)
        else:
            if missing_entire_obs:
                design = np.zeros(model.design.shape[:-1])
            else:
                design = model.design[:, :, design_t]
            obs_cov = model.obs_cov[:, :, obs_cov_t]
            kalman_gain = kfilter.kalman_gain[:, :, t]
            forecasts_error_cov = kfilter.forecasts_error_cov[:, :, t]
            forecasts_error = kfilter.forecasts_error[:, t]
            F_inv = np.linalg.inv(forecasts_error_cov)

        # Create a temporary matrix
        tmp_L[:, :, t] = transition - kalman_gain.dot(design)
        L = tmp_L[:, :, t]

        # Perform the recursion

        # Intermediate values
        if smoother_output & (SMOOTHER_STATE | SMOOTHER_DISTURBANCE):
            if missing_entire_obs:
                # smoothing_error is undefined here, keep it as zeros
                scaled_smoothed_estimator[:, t - 1] = (
                    transition.transpose().dot(scaled_smoothed_estimator[:, t])
                )
            else:
                smoothing_error[:k_endog, t] = (
                    F_inv.dot(forecasts_error) -
                    kalman_gain.transpose().dot(
                        scaled_smoothed_estimator[:, t])
                )
                scaled_smoothed_estimator[:, t - 1] = (
                    design.transpose().dot(smoothing_error[:k_endog, t]) +
                    transition.transpose().dot(scaled_smoothed_estimator[:, t])
                )
        if smoother_output & (SMOOTHER_STATE_COV | SMOOTHER_DISTURBANCE_COV):
            if missing_entire_obs:
                scaled_smoothed_estimator_cov[:, :, t - 1] = (
                    L.transpose().dot(
                        scaled_smoothed_estimator_cov[:, :, t]
                    ).dot(L)
                )
            else:
                scaled_smoothed_estimator_cov[:, :, t - 1] = (
                    design.transpose().dot(F_inv).dot(design) +
                    L.transpose().dot(
                        scaled_smoothed_estimator_cov[:, :, t]
                    ).dot(L)
                )

        # State smoothing
        if smoother_output & SMOOTHER_STATE:
            smoothed_state[:, t] = (
                predicted_state +
                predicted_state_cov.dot(scaled_smoothed_estimator[:, t - 1])
            )
        if smoother_output & SMOOTHER_STATE_COV:
            smoothed_state_cov[:, :, t] = (
                predicted_state_cov -
                predicted_state_cov.dot(
                    scaled_smoothed_estimator_cov[:, :, t - 1]
                ).dot(predicted_state_cov)
            )

        # Disturbance smoothing
        if smoother_output & (SMOOTHER_DISTURBANCE | SMOOTHER_DISTURBANCE_COV):
            QR = state_cov.dot(selection.transpose())

        if smoother_output & SMOOTHER_DISTURBANCE:
            smoothed_state_disturbance[:, t] = (
                QR.dot(scaled_smoothed_estimator[:, t])
            )
            # measurement disturbance is set to zero when all missing
            # (unconditional distribution)
            if not missing_entire_obs:
                smoothed_measurement_disturbance[mask, t] = (
                    obs_cov.dot(smoothing_error[:k_endog, t])
                )

        if smoother_output & SMOOTHER_DISTURBANCE_COV:
            smoothed_state_disturbance_cov[:, :, t] = (
                state_cov -
                QR.dot(
                    scaled_smoothed_estimator_cov[:, :, t]
                ).dot(QR.transpose())
            )

            if missing_entire_obs:
                smoothed_measurement_disturbance_cov[:, :, t] = obs_cov
            else:
                # For non-missing portion, calculate as usual
                ix = np.ix_(mask, mask, [t])
                smoothed_measurement_disturbance_cov[ix] = (
                    obs_cov - obs_cov.dot(
                        F_inv + kalman_gain.transpose().dot(
                            scaled_smoothed_estimator_cov[:, :, t]
                        ).dot(kalman_gain)
                    ).dot(obs_cov)
                )[:, :, np.newaxis]

                # For missing portion, use unconditional distribution
                ix = np.ix_(~mask, ~mask, [t])
                mod_ix = np.ix_(~mask, ~mask, [0])
                smoothed_measurement_disturbance_cov[ix] = np.copy(
                    model.obs_cov[:, :, obs_cov_t:obs_cov_t+1])[mod_ix]

        # Advance the smoother
        self.t -= 1
