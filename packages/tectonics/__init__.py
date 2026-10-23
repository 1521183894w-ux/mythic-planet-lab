"""构造模拟（阶段 1，路线 B：剧本型运动学 + 共用地壳规则）。方法说明见 docs/tectonics_method.md。

模块：
- geometry.py    球面几何与旋转；
- mesh.py        stripy 二十面体网格（缓存在 .cache/tectonics/）与栅格化；
- crust.py       地块单元表、参数、构造尺度高程（GDH1 洋深、Airy 均衡）；
- engine.py      地壳引擎：板块参照系中的地块单元、俯冲、碰撞、扩张、裂解、地体转移；
- kinematics.py  板块运动剧本、PLATES4 旋转文件、pyGPlates 逐步阶段旋转；
- synthetic.py   合成试验（裂谷 -> 洋盆扩张、碰撞 -> 造山）；
- prehistory.py  前史：300 Ma 初始大陆（省份拼贴 -> 沿薄弱带裂解 -> 漂移）；
- emergent.py    涌现型运动学（路线 A，只用于前史）；
- reverse.py     倒推板块运动：从终图出发的关键时刻有限旋转、插值、层级组合、速度（U22）。
"""
