import pytest
from blase.training.defaults.registry import compiler, fitter, wrap_existing_dataset

tf = pytest.importorskip("tensorflow")


def _tiny_model():
    inp = tf.keras.Input((4,))
    x = tf.keras.layers.Dense(2, activation="relu")(inp)
    out = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inp, out)


def test_compile_and_fit_numpy():
    x = tf.random.uniform((32, 4))
    y = tf.random.uniform((32, 1))
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(8)
    builder = wrap_existing_dataset(ds, ds, None, assume_batched=True)
    train_ds, val_ds, _ = builder.build()

    mdl = _tiny_model()
    comp = compiler("tensorflow")
    comp.compile(
        mdl,
        tf.keras.optimizers.Adam(1e-3),
        tf.keras.losses.MeanSquaredError(),
        [tf.keras.metrics.MeanAbsoluteError(name="mae")],  # or just "mae"
        lr_schedule=None,
    )

    fit = fitter("tensorflow")
    res = fit.fit(
        mdl,
        train_ds,
        val_ds,
        epochs=1,
        steps_per_epoch=2,
        validation_steps=1,
        class_weight=None,
        callbacks=[],
    )
    assert "history" in res and "model" in res
