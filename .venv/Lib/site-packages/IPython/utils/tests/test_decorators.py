from IPython.utils import decorators

def test_flag_calls():
    @decorators.flag_calls
    def f():
        pass
    
    assert not f.called
    f()
    assert f.called