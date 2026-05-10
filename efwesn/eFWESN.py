"""
@created:
Mon Feb 16 2026

@description:
The Fuzzy Weighted Echo State Model is a hybrid predictive model proposed by Yao Zhao and Yingshun Li.
This script transforms the original proposal, README [1], into an evolving model using
    - evolving membership
    - recursive least squares
"""

import numpy
import pandas


class eFWESN:
    def __init__(self,
                 dim_in: int,       # dimension for X 
                 dim_res: int,      # dimension per Γ
                 dim_out:int,       # dimension for Y
                 cauchy_r: float,   # cauchy radius
                 firing_th: float,  # μ, firing strenth threshold
                 spectral_r: float, # spectral radius
                 init_cov:float=1000,   # initial covariance
                 washout_period:float=100   # washout period
                ) -> None:

        self.hyperparameters = pandas.DataFrame({
            '#x': [dim_in],     # dimension for x
            '#Γ': [dim_res],    # dimension per Γ
            '#y': [dim_out],    # dimension for y
            'cauchy_r': [cauchy_r], # cauchy radius
            '∀μ_th': [firing_th],    # raw firing threshold
            '∀ρ_Γ': [spectral_r],     # rule spectral radius
            'C0_Γ': [init_cov],    # rule initial covariance
            'τ_washout': [washout_period]   # washout period for initial data
        })

        # placeholder input array
        # input,
        # feedback, (predict or real value)
        self.x = [
            None,   # in
            None,   # feedback
        ]
        # input weight matrix and its covariance
        self.w_in = numpy.zeros(shape=(dim_in, dim_out))
        self.C_in = init_cov * numpy.eye(dim_in)
        # feedback weight matrix and its covariance
        self.w_fb = numpy.zeros(shape=(dim_out, dim_out))
        self.C_fb = init_cov * numpy.eye(dim_out)
        # fuzzy weighted esn
        self.rules = pandas.DataFrame(columns=[
            # membership
            'center',   # rule origin
            'μ',    # rule strength value
            'λ',    # rule strength normalized
            # reservoir
            'w_in',     # input weight
            'w_fb',   # feedback weight
            'w_r',        # reservoir weight
            'Γ',        # reservoir state
            # output
            'w',    # weight applied to this reservior
            'C'     # covariance of this reservior
        ])
        # expected output array:
        # model output
        # real output
        self.y = [
            numpy.zeros(shape=dim_out).reshape((1, -1)),    # model
            None                                            # real
        ]

    
    # ! PIPE
    

    def run(self, X:numpy.ndarray, Y: None | numpy.ndarray = None) -> numpy.ndarray:
        """
        pass input
        get predicted output

        It flows in the order:
        + μ(X), Membership of the input to every rule
        + λ(μ), Normalized firing rate based on memberships
        + Γ(X), Reserviour state based on X:input, Γ:reserviour and Y:output
        + Y(X, Y, Γ), Output of the system
        """

        # Reshape for matrix operations
        self.x = [
            X.reshape((1, -1)), # in
            self.y[1] or self.y[0]  # feedback (real or model)
        ]
        self.y = [
            None,  # predict
            None if Y is None else Y.reshape((1, -1))
        ]
        
        # * ------------------------------------
        # * Firing Strengths, μ(X)
        # * ------------------------------------
        # * The membership of a rule utilizes a Cauchy function.
        # * It is applied to every dimension of a point.
        # * The firing strength is based on all dimensions of a point.
        # *
        # *                                  1
        # * Cauchy, cX =  ------------------------------------
        # *                   1 + (2 * (X - center) / r)^2
        # * μX = ∏i cX
        # * ------------------------------------
        self.rules_firing_strengths()

        # * ------------------------------------
        # * Rule Manage
        # * ------------------------------------
        # * Rule generation criterion is firing strength.
        # * A new rule is initialized by the current input, X,
        # * iff the max firing strength of all rules is less than threshold, ∀μ_th.
        # * ------------------------------------
        self.rules_manage()

        # * ------------------------------------
        # * Normalized Firing Strengths, λ(X)
        # * ------------------------------------
        # * Normalized the firing strength for contribution factor
        # *
        # * λX = μX_{i} / ∑i μX_{i}
        # * ------------------------------------
        self.rules_norm_firing_strengths()

        # * ------------------------------------
        # * Echo State Path, Γ(X)
        # * ------------------------------------
        # * The input is passed to the reservoirs of each rule
        # * The new state of the reserviour is based on the
        # *     - input
        # *     - previous output
        # *     - previous state
        # * 
        # * Rather than the leaky integration:
        # * x_{t+1} = (1-α)x_{t} + αϕ(W_{res}x_{t} + W_{in}u_{t+1} + b)
        # * 
        # * ΓX = ϕ(X@W_{in} + Γ@W + Y_{t-1}@W_{back})
        # * ------------------------------------
        self.reserviour_state_update()
        
        
        # * ------------------------------------
        # * Combined Output, y(X)
        # * ------------------------------------
        # * The output feed combines the states of
        # *     - input
        # *     - output
        # *     - reserviour
        # * 
        # * Y_{X,Y,Γ} = X@W_{in} + Y@W_{in} + ∑ λ·Γ@ W_{Γ}
        # * ------------------------------------

        # initialize with the direct path from input and feedbacka
        self.y[0] = (
            self.x[0] @ self.w_in +    # input contribution
            self.x[1] @ self.w_fb   # feedback contribution 
        )
        for row in self.rules.index:
            # apply the membership of the rule
            self.y[0] = self.y[0] + (
                self.rules.loc[row, 'λ'] *  # contribution factor
                self.rules.loc[row, 'Γ'] @  # reservoir state
                self.rules.loc[row, 'w']    # weight matrix
            )

        # * ------------------------------------
        # * Consequent Parameters
        # * ------------------------------------
        # * Update weights with covariance matrices using Recursive Least Squares
        # * Updates the feedback(y) to the real value
        # * 
        # * Only the W_{out} is trained
        # *     W_{out} consists of
        # *         - weights for the input
        # *         - weights for the feedback
        # *         - weights from the reservior, Γ
        # * ------------------------------------
        if Y is not None:
            # update input covariance/weight
            self.C_in = self.rls_covariance({
                'λ': 1,
                'Γ': self.x[0],
                'C': self.C_in
            })
            self.w_in = self.rls_weight({
                'λ': 1,
                'Γ': self.x[0],
                'Y': self.y[1],
                'C': self.C_in,
                'W': self.w_in
            })

            # update feedback covariance/weight
            self.C_fb = self.rls_covariance({
                'λ': 1,
                'Γ': self.x[1],
                'C': self.C_fb
            })
            self.w_in = self.rls_weight({
                'λ': 1,
                'Γ': self.x[1],
                'Y': self.y[1],
                'C': self.C_fb,
                'W': self.w_fb
            })

            # update rules covariance/weight
            for row in self.rules.index:
                self.rules.at[row, 'C'] = self.rls_covariance({
                    'λ': self.rules.loc[row, 'λ'],
                    'Γ': self.rules.loc[row, 'Γ'],
                    'C': self.rules.loc[row, 'C']
                })
                self.rules.at[row, 'w'] = self.rls_weight({
                    'λ': self.rules.loc[row, 'λ'],
                    'Γ': self.rules.loc[row, 'Γ'],
                    'Y': self.y[1],
                    'C': self.rules.loc[row, 'C'],
                    'W': self.rules.loc[row, 'w']
                })

        return self.y[0]



    # ! SUPPORT

    def rules_manage(self) -> None:
        """
        creates a new rule
        lengthens the W_{out} matrix
        
        :param self: Description
        :param X: Description
        """

        # Condition for Initializing a New Rule
        array_u = self.rules['μ']
        if not array_u.empty and max(array_u) >= self.hyperparameters.loc[0, '∀μ_th']:
            return

        sr = self.hyperparameters.loc[0, '∀ρ_Γ']
        dim_in = self.hyperparameters.loc[0, '#x']
        dim_res = self.hyperparameters.loc[0, '#Γ']
        dim_out = self.hyperparameters.loc[0, '#y']
        gen = numpy.random.default_rng(self.rules.shape[0])

        # Generate the reserviour matrices
        W_in = gen.uniform(-1, 1, (dim_in, dim_res))
        W_fb = gen.uniform(-1, 1, (dim_out, dim_res))
        # Use abs because eigenvalue, to get the spectral radius, might be a complex value
        # Normalize with the intrinsic spectral radius and apply the target spectral radius
        W_r = gen.uniform(-1, 1, (dim_res, dim_res))
        W_r = W_r * (sr / numpy.max(numpy.abs(numpy.linalg.eigvals(W_r))))

        # Update the rules
        rule = pandas.DataFrame([{
            'center': self.x[0],  # center
            'μ': 0,
            'λ': 0,
            'w_in': W_in,
            'w_fb': W_fb,
            'w_r': W_r,
            'Γ': numpy.zeros(shape=dim_res).reshape((1, -1)),   # reservoir state initialized to 0
            'w': numpy.zeros(shape=(dim_res, dim_out)), # initial output weight for the reserviour
            'C': self.hyperparameters.loc[0, 'C0_Γ'] * numpy.eye(dim_res) # initial weight covariance for the reserviour
        }])
        self.rules = pandas.concat(
            [self.rules, rule],
            ignore_index=True
        )

        # * Update all firing strengths
        # re calculate the Cauchy memberships
        self.rules_firing_strengths()

    
    @staticmethod
    def rls_covariance(kwargs: dict):
        """
        kwargs must contain:
            - 'λ'     update scale
            - 'C'     covariance
            - 'Γ'     input (usually reservior but also input and feedback)
        
        updates the covariance matrix

                 λ * C @ Γ @ Γ.T @ C
        C = C - ---------------------
                 1 + λ * Γ.T @ C @ Γ
        """

        numerator = kwargs['λ'] * kwargs['C'] @ kwargs['Γ'].T @ kwargs['Γ'] @ kwargs['C']
        dumenator = 1 + kwargs['λ'] * kwargs['Γ'] @ kwargs['C'] @ kwargs['Γ'].T
        
        return kwargs['C'] - (numerator / dumenator)


    @staticmethod
    def rls_weight(kwargs: dict):
        """
        kwargs must contain:
            - 'λ'   update scale
            - 'C'   covariance
            - 'Γ'   input (usually reservior but also input and feedback)
            - 'W'   output weight
            - 'Y'   actual output

        W = W + (C @ Γ * λ) * (Y - Γ @ W)

        updates the associated weight
        """

        return kwargs['W'] + (kwargs['C'] @ kwargs['Γ'].T * kwargs['λ']) * (kwargs['Y'] - kwargs['Γ'] @ kwargs['W'])


    def rules_firing_strengths(self) -> None:
        """
        calculates the cauchy memberships of the point to all existing rules
        
                                         1
        Cauchy, cX =  ------------------------------------
                          1 + ((X - center) / r)^2
        μX = ∏i cX
        """

        for row in self.rules.index:
            # membership from every rule
            self.rules.at[row, 'μ'] = 1 / numpy.prod(
                1 + numpy.pow((self.x[0] - self.rules.loc[row, 'center']) / self.hyperparameters.loc[0, 'cauchy_r'], 2)
            )

        
    def rules_norm_firing_strengths(self) -> None:
        """
        normalized the cauchy memberships of all existing rules

        λX = μX_{i} / ∑i μX_{i}
        """
        # summation of raw firing strengths
        total_u = sum(self.rules['μ'])

        # calculate the normalized membership
        for row in self.rules.index:
            self.rules.at[row, 'λ'] = self.rules.loc[row, 'μ'] / total_u


    def reserviour_state_reset(self):
        """
        resets reserviour states to zero
        resets reserviour washout to > 1
        """

        for row in self.rules.index:
            self.rules.at[row, 'Γ'] = numpy.zeros(
                shape=self.rules.loc[row, 'Γ'].shape
            )
        self.hyperparameters.at[0, 'τ_washout'] = 100
    

    def reserviour_state_update(self) -> None:
        """
        updates the reserviour state relative to
            - input
            - previous output
            - previous state
        
        ΓX = ϕ(X@W_{in} + Γ@W + Y_{t-1}@W_{back})
        """

        for row in self.rules.index:
            for _ in range(self.hyperparameters.loc[0, 'τ_washout']):
                # update reservoir state for every rule
                self.rules.at[row, 'Γ'] = numpy.tanh(
                    self.x[0] @ self.rules.loc[row, 'w_in'] +
                    self.x[1] @ self.rules.loc[row, 'w_fb'] +
                    self.rules.loc[row, 'Γ'] @ self.rules.loc[row, 'w_r']
                )
        # reset washout to 1
        self.hyperparameters.loc[0, 'τ_washout'] = 1


