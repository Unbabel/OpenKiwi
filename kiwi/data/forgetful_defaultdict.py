class ForgetfulDefaultdict(dict):
    """Defaultdict that does not cache values.
    """
    def __init__(self, default, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default = default

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        else:
            return self.default
