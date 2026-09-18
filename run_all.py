# -*- coding: utf-8 -*-
"""便捷入口，等价于 `python -m aaa_isp_lab`。

保留这个文件是为了让**没安装包**的人也能直接 `python run_all.py` 跑起来。
正式入口是命令行脚本 `aaa-isp-lab`（`pip install -e .` 之后可用）。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from aaa_isp_lab.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
