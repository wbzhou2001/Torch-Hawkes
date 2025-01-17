class LambdaScheduler:
    '''
    Helper class
    For unstationary forecast whose future value will grow larger, which needs to tune "up" the lam_bar for thinning algorithm
    '''
    def __init__(self,
                 init_val   = 1e+2,
                 add_val    = 1e+2,
                 minus_val  = 1e+1,
                 patience   = 100,
                 verbose    = False):
        '''
        Args:
        - init_val  : initial value of the lambda
        - add_val   : increment unit of lambda
        - minus_val : decrement unit of lambda
        - patience  : rounds of success before lambda is decreased
        '''
        assert add_val > minus_val, f"LambdaScheduler: Expected add_val > minus_val, got {add_val} <= {minus_val}!"
        self.init_val   = init_val
        self.patience   = patience
        self.add_val    = add_val
        self.minus_val  = minus_val
        self.verbose    = verbose
        # init params
        self.lam_bar    = self.init_val
        self.patience_count = 0

    def __call__(self):
        return self.lam_bar

    def step(self, flag):
        '''
        Args:
        - flag : boolean. True - a point has been successfully simulated.
        '''
        if flag:
            if self.patience_count >= self.patience:
                self.patience_count = 0 
                self.lam_bar -= self.minus_val
                if self.verbose:
                    print(f'Decreasing lam_bar from {self.lam_bar + self.minus_val} to {self.lam_bar}')
            else:
                pass
            self.patience_count += 1
        else:
            self.patience_count = 0
            self.lam_bar += self.add_val
            if self.verbose:
                print(f'Increasing lam_bar from {self.lam_bar - self.add_val} to {self.lam_bar}')

    def reset(self):
        self.lam_bar = self.init_val