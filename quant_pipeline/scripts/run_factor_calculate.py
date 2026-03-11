# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 09:34:22 2026

@author: jimya
"""

"""
Purpose：全量计算历史因子库（可并行/分块）。
TODO：读 config → load raw/processed → factor registry 批量执行 → 写 parquet → 生成 manifest（因子、窗口、日期范围、版本）。
"""