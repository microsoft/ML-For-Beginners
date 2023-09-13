class T:

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return self.name


td = {T("foo", 24): "bar",
      T("gad", 42): "zooks",
      T("foo", 12): "bur"}

print('TEST SUCEEDED!')  # Break here
