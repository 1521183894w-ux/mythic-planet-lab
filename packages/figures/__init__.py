"""出图工具（地球基准与虚构世界共用）与标准图组管线（scripts/make_figures.py）。

模块：
- world.py           世界读取（地球格网 / 虚构世界 netCDF）、地球专用设置、JSON 转换；
- style.py           统一色带与山体阴影、中文字体、比例尺、图底说明（method_class、种子、代码版本）；
- regional.py        局部等面积方位投影窗口、经纬网、区域面板；
- select_regions.py  8 张区域放大图的自动选取（review_protocol §2.2）；
- maps.py            全球图、极地图、指标分布图；
- charts.py          与同分辨率地球的对比图；
- region_panels.py   区域放大图（世界 | 地球对照）；
- gallery.py         index.html 画廊。
"""
