"""构造模拟（阶段 1，路线 B：剧本型运动学 + 共用地壳规则）。方法说明见 docs/tectonics_method.md。

模块：
- geometry.py    球面几何与旋转；
- mesh.py        stripy 二十面体网格（缓存在 .cache/tectonics/）与栅格化；
- crust.py       地块单元表、参数、构造尺度高程（GDH1 洋深、Airy 均衡）；
- engine.py      地壳引擎：板块参照系中的地块单元、俯冲、碰撞、扩张、裂解、地体转移；
- kinematics.py  板块运动剧本、PLATES4 旋转文件、pyGPlates 逐步阶段旋转。
"""
