from coflect.modules.hitl.launcher import RunConfig, _build_process_specs, _server_host_for_clients


def _base_cfg(**overrides: object) -> RunConfig:
    base = {
        "backend": "torch",
        "host": "0.0.0.0",
        "port": 8000,
        "steps": 1000,
        "xai_every": 100,
        "forecast_every": 20,
        "snapshot_every": 200,
        "batch_size": 64,
        "trainer_device": "",
        "xai_device": "",
        "dataset": "cifar10_catsdogs",
        "data_root": "./data",
        "split": "train",
        "download_data": True,
        "num_workers": 0,
        "xai_method": "consensus",
        "xai_budget_ms_per_minute": 1234.0,
        "snapshot_dir": "",
        "log_dir": "./.coflect_logs/hitl",
        "startup_wait_s": 1.5,
    }
    base.update(overrides)
    return RunConfig(**base)


def test_server_host_mapping() -> None:
    assert _server_host_for_clients("0.0.0.0") == "127.0.0.1"
    assert _server_host_for_clients("::") == "127.0.0.1"
    assert _server_host_for_clients("localhost") == "localhost"


def test_build_specs_torch_real_data() -> None:
    specs = _build_process_specs(_base_cfg())
    names = [s.name for s in specs]
    assert names == ["backend", "forecast", "xai_worker", "trainer"]

    xai = next(s for s in specs if s.name == "xai_worker")
    trainer = next(s for s in specs if s.name == "trainer")
    assert xai.module.endswith("worker_torch_livecam")
    assert trainer.module.endswith("train_torch")
    assert "--download_data" in xai.args
    assert "--download_data" in trainer.args
    assert xai.args[xai.args.index("--snapshot_dir") + 1] == "snapshots"
    assert trainer.args[trainer.args.index("--snapshot_dir") + 1] == "snapshots"


def test_build_specs_tensorflow_snapshot_dir() -> None:
    specs = _build_process_specs(_base_cfg(backend="tensorflow", snapshot_dir="custom_tf_snap"))
    xai = next(s for s in specs if s.name == "xai_worker")
    trainer = next(s for s in specs if s.name == "trainer")
    forecast = next(s for s in specs if s.name == "forecast")
    assert xai.module.endswith("worker_tf_livecam")
    assert trainer.module.endswith("train_tf")
    assert "--dataset" not in trainer.args
    assert "--data_root" not in xai.args
    assert trainer.args[trainer.args.index("--snapshot_dir") + 1] == "custom_tf_snap"
    assert xai.args[xai.args.index("--snapshot_dir") + 1] == "custom_tf_snap"
    assert forecast.args[forecast.args.index("--backend") + 1] == "tensorflow"
