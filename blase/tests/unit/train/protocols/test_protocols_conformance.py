from typing import Sequence, Optional, Mapping, Any
from blase.training.protocols import (
    DatasetBuilder,
    StrategyAdapter,
    PrecisionPolicy,
    Compiler,
    Fitter,
    Evaluator,
    Exporter,
    CallbackFactory,
    Tracker,
)


class _DB(DatasetBuilder):
    def build(self):
        return ([], [], [])


class _Strat(StrategyAdapter):
    def scope(self):
        class _Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _Ctx()

    def shard_dataset(self, ds, is_train: bool):
        return ds


class _Prec(PrecisionPolicy):
    def apply(self) -> None:
        pass


class _Comp(Compiler):
    def compile(
        self, model, optimizer, loss, metrics: Sequence[Any], lr_schedule, jit=None
    ):
        return model


class _Fit(Fitter):
    def fit(
        self,
        model,
        train,
        val,
        epochs,
        spe,
        vsteps,
        class_weight: Optional[Mapping[Any, float]],
        callbacks: Sequence[Any],
    ):
        return {"history": {}, "model": model}


class _Eval(Evaluator):
    def evaluate(self, model, ds, steps: Optional[int]):
        return {"loss": 0.0}


class _Exp(Exporter):
    def export(self, model, kind: str, path: Optional[str], extra=None) -> str:
        return path or "out"


class _CB(CallbackFactory):
    def build(self) -> Sequence[Any]:
        return []


class _Tr(Tracker):
    def start_step(self, fqn: str, params):
        class _Ctx:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def log_metric(self, *a, **k):
                pass

            def log_artifact(self, *a, **k):
                pass

            @property
            def params(self):
                return {}

        return _Ctx()

    def log_metric(self, *a, **k):
        pass

    def log_artifact(self, *a, **k):
        pass

    def end_step(self, status: str = "ok"):
        pass


def test_protocol_min_impls():
    _DB().build()
    entered = False
    with _Strat().scope():
        entered = True
    assert entered
    _Prec().apply()
    _Comp().compile(object(), object(), object(), [], None)
    _Fit().fit(object(), [], [], 1, None, None, None, [])
    _Eval().evaluate(object(), [], None)
    assert _Exp().export(object(), "saved_model", None).endswith("out")
    assert _CB().build() == []
    _Tr().start_step("x", {}).log_metric("m", 1)
