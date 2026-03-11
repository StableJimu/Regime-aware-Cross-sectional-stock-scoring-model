# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 09:35:13 2026

@author: jimya
"""

"""
Purpose：增量更新最近 N 天因子 + 自动对齐依赖窗口。
TODO：检测已有数据最新日期 → 补齐 raw → 重新计算涉及窗口的尾部区间 → append/overwrite 分区。
"""