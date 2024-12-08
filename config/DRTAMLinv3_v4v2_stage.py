# -*- coding: UTF-8 -*-


plugins=[
        dict(cfg=dict(type='DRTAMLinv3', hw_chunk = 3, c_chunk = 1, reduction_ratio=16, dual_artecture = True,
                 c_pool_list=['AvgChunkPool','MaxChunkPool'],
                 hw_pool_list=['AvgChunkPool','MaxChunkPool'],
                 hwc_pool_list=['AvgChunkPool','MaxChunkPool']),
             stages=(True, True, True, True, True),
             position='stage_output')
        ]