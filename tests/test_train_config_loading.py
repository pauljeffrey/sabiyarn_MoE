from pathlib import Path

from training.load_config import _normalize_list_sections, load_train_config


def test_load_train_config_reads_yaml_sections():
    cfg = load_train_config(Path('training/train_config.yaml'))
    assert cfg.mode == 'pretrain'
    assert cfg.model_name == 'Aletheia-ng/SabiYarn_MoE_translate_base'
    assert cfg.train_data_paths == ['datasets/eng_training.bin', 'datasets/training.bin']
    assert cfg.process_one_file_at_a_time is True


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
