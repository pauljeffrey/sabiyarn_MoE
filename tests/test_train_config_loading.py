from pathlib import Path

from training.load_config import load_train_config


def test_load_train_config_reads_yaml_sections():
    cfg = load_train_config(Path('training/train_config.yaml'))
    assert cfg.mode == 'pretrain'
    assert cfg.model_name == 'Aletheia-ng/SabiYarn_MoE_translate_base'
    assert cfg.train_data_paths == ['datasets/eng_training.bin', 'datasets/training.bin']
    assert cfg.eval_data_path == 'validation.bin'
    assert cfg.process_one_file_at_a_time is True
