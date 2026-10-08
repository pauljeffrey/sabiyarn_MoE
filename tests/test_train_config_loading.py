from pathlib import Path

from training.load_config import _normalize_list_sections, load_train_config


def test_load_train_config_reads_yaml_sections():
    cfg = load_train_config(Path('training/train_config.yaml'))
    assert cfg.mode == 'pretrain'
    # model_name (model.repo_name) and the data paths below are under active
    # tuning -- assert structure, not literal values.
    assert cfg.model_name.startswith('Aletheia-ng/SabiYarn_MoE')
    assert len(cfg.train_data_paths) == 3  # [eng, afr, other_afr]
    assert cfg.train_data_paths[2].endswith('other_african_pretrain.bin')
    assert all(p.startswith('datasets/') and p.endswith('.bin') for p in cfg.train_data_paths)
    assert cfg.process_one_file_at_a_time is True


def test_use_loss_mask_and_last_step_read_from_yaml(tmp_path):
    yaml_text = (
        "training:\n"
        "  use_loss_mask: false\n"
        "  last_step: 1700\n"
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml_text)
    cfg = load_train_config(path)
    assert cfg.use_loss_mask is False
    assert cfg.last_step == 1700


def test_use_loss_mask_defaults_true_when_absent(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("training:\n  mode: pretrain\n")
    cfg = load_train_config(path)
    assert cfg.use_loss_mask is True


def test_last_step_defaults_to_none_when_absent(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("training:\n  mode: pretrain\n")
    cfg = load_train_config(path)
    assert cfg.last_step is None


def test_last_step_blank_string_treated_as_none(tmp_path):
    yaml_text = "training:\n  last_step: \"\"\n"
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml_text)
    cfg = load_train_config(path)
    assert cfg.last_step is None


def test_resume_from_read_from_yaml(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("training:\n  resume_from: best\n")
    cfg = load_train_config(path)
    assert cfg.resume_from == "best"


def test_resume_from_defaults_to_latest_when_absent(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("training:\n  mode: pretrain\n")
    cfg = load_train_config(path)
    assert cfg.resume_from == "latest"


def test_reference_model_repo_read_from_yaml(tmp_path):
    yaml_text = (
        "model:\n"
        "  reference_repo: \"Aletheia-ng/sabiyarn-ref\"\n"
        "training:\n"
        "  reference_weight_deviation_threshold: 0.05\n"
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml_text)
    cfg = load_train_config(path)
    assert cfg.reference_model_repo == "Aletheia-ng/sabiyarn-ref"
    assert cfg.reference_weight_deviation_threshold == 0.05


def test_reference_model_repo_defaults_to_none_when_absent(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("training:\n  mode: pretrain\n")
    cfg = load_train_config(path)
    assert cfg.reference_model_repo is None
    assert cfg.reference_weight_deviation_threshold == 0.5


def test_normalize_list_sections_tolerates_comment_lines_mid_block():
    text = (
        "data:\n"
        "  pretrain:\n"
        "    - eng_train_data_path: \"a.bin\"\n"
        "    # a standalone comment shouldn't terminate the list block\n"
        "    - eval_data_path: \"b.bin\"\n"
    )
    import yaml
    parsed = yaml.safe_load(_normalize_list_sections(text))
    assert parsed["data"]["pretrain"]["eng_train_data_path"] == "a.bin"
    assert parsed["data"]["pretrain"]["eval_data_path"] == "b.bin"


def test_ddp_enabled_selects_strategy(monkeypatch):
    from training.load_config import _resolve_distributed
    monkeypatch.delenv("DISTRIBUTED", raising=False)
    assert _resolve_distributed({"enabled": True}, {}) == "ddp"
    assert _resolve_distributed({"enabled": False}, {"distributed": "ddp"}) == "fsdp"
    assert _resolve_distributed({}, {"distributed": "fsdp"}) == "fsdp"   # legacy key still honoured
    assert _resolve_distributed({}, {}) == "ddp"
    monkeypatch.setenv("DISTRIBUTED", "FSDP")
    assert _resolve_distributed({"enabled": True}, {}) == "fsdp"          # env wins


def test_yaml_default_is_ddp_bf16(monkeypatch):
    monkeypatch.delenv("DISTRIBUTED", raising=False)
    cfg = load_train_config("training/train_config.yaml")
    assert cfg.distributed == "ddp" and cfg.mixed_precision in ("bf16", "bfloat16")


def test_sft_phase_resizes_model_pretrain_does_not(monkeypatch):
    monkeypatch.delenv("TRAIN_MODE", raising=False)
    pre = load_train_config(Path('training/train_config.yaml'))
    assert (pre.resize_vocab_size, pre.resize_block_size, pre.fill_positions_from) == (0, 0, 0)       # continuous pretraining never grows the model
    assert pre.block_size == 4096
    monkeypatch.setenv("TRAIN_MODE", "sft")
    sft = load_train_config(Path('training/train_config.yaml'))
    assert sft.resize_vocab_size == 52160 and sft.resize_vocab_size % 64 == 0 and sft.resize_vocab_size >= 52117
    assert sft.resize_block_size == 0 and sft.block_size == 32768                                    # no position-table growth unless asked for
    assert sft.train_batch_size * sft.gradient_accumulation_steps * sft.block_size == 786_432


def test_sft_masks_the_loss_even_though_pretraining_does_not(monkeypatch):
    monkeypatch.delenv("TRAIN_MODE", raising=False)
    assert load_train_config(Path('training/train_config.yaml')).use_loss_mask is False
    monkeypatch.setenv("TRAIN_MODE", "sft")
    assert load_train_config(Path('training/train_config.yaml')).use_loss_mask is True
