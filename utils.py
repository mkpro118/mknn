import sys as _sys


def export_function(fn):
    mod = _sys.modules[fn.__module__]
    _ = fn.__name__
    if hasattr(mod, '__all__'):
        mod.__all__.append(_)
    else:
        mod.__all__ = [_]
    if hasattr(mod, 'exports'):
        mod.exports[_] = fn
    else:
        mod.exports = {_: fn}
    return fn


def export_class(cls):
    mod = _sys.modules[cls.__module__]
    _ = cls.__name__
    if hasattr(mod, '__all__'):
        mod.__all__.append(_)
    else:
        mod.__all__ = [_]
    if hasattr(mod, 'exports'):
        mod.exports[_] = cls
    else:
        mod.exports = {_: cls}
    return cls
