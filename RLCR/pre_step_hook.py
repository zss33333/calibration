"""Call an observer just before an engine's optimizer step."""


def install_pre_step_hook(engine, observer):
    original_step = engine.step

    def observed_step(*args, **kwargs):
        observer()
        return original_step(*args, **kwargs)

    engine.step = observed_step
    return original_step
