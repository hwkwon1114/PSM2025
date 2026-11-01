"""Compatibility entry point; shared exporter preserves pass order and overwrite protection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shape_design.export_illustrated import SOURCE, main

if __name__ == '__main__':
    main(default_k=-4, default_with_shape=True, default_out=SOURCE / 'k_-4_illustrated_with_shape')
