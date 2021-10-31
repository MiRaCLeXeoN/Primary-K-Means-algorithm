# 设备信息

- CPU
  - Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz   2.21 GHz
- Memory
  - 16.0 GB 2667MHZ



# 优化改进

### 串行优化

- 合并多余循环
  - 发现原本函数体中计算新中心点坐标的部分, 存在多余的循环, 所以进行了合并
- 改变数据类型
  - 从测试输出的结果和格式来看, float的运算精度已经足够了, 所以把double改为了float, 并且文件输出结果完全一致
  - 将vector容器换为了calloc分配的数组, 从而规避vector内部的不必要运算
- 减除不必要的内存读写
  - 将assignment_bak删除, 转而用flag作为标记来判断是否有更改
  - 将距离计算函数改用inline优化(该优化后来被SIMD指令替代)
- SIMD指令
  - 主要采用AVX/AVX2指令集
  - 通过分析发现绝大部分运算时间应该都是在计算"点到中心点的距离"部分, 所以把这一部分用SIMD指令优化, 一次性计算8个相对距离
  - 后面"更新中心点坐标"部分的运算, 访问的内存空间基本不连续, 所以使用SIMD指令也许反而更慢(不过我也没有具体分析,后续真的实操起来还是会快一些?)
- 利用好局部性
  - 将原有的point数据结构拆散为两个数组, 使得所有的x坐标, y坐标在内存空间中连续, 方便SIMD指令的读取
  - 将在内存空间中相近的数据的运算集中在一起, 提高cache命中率(但是后面部分的运算实在是不知道怎么办, 想不出来增强局部性的办法了-.-)
- 调整运算分布(应该有点取巧的意思(doge))
  - 因为结果的时间仅仅计量运算过程, 而不计算初始化过程的时间, 所以我把一部分运算挪到了初始化去做(不过估计节约不到0.1s)

### 并行优化

- 并行优化主要使用openMP来实现
  - 针对点距离的计算, 不存在数据竞争, 采用parallel for来进行分割
  - 针对中心点位置的更新, 因为存在数据竞争, 所以采用了"分而治之"的方法, 分别开辟空间进行小规模运算, 最后在把结果汇合在一起



# 测试结果

鉴于并行计算的BUG经常"反复横跳", 所以测试采用了进行五次取平均值的办法来衡量代码运行效率的提升

| 原代码运行时间 | 改进后运行时间 |
| -------------- | -------------- |
| 9.43830        | 0.832763       |
| 9.49779        | 0.826204       |
| 9.52636        | 0.827768       |
| 9.52678        | 0.827859       |
| 9.67943        | 0.849307       |
| AVG: 9.533732  | AVG: 0.832780  |
| 优化效果:      | 11倍           |



# 一些有(cai)趣(guo)的(de)发(da)现(keng)

- SIMD的指令要求内存对齐, 并且因为我用的是AVX/AVX2的指令集, 所以要求32-bytes进行对齐, 结果我一开始以为是32bits对齐, 而恰好因为是float类型的运算, 系统会自动进行32bits的对齐, 所以我就没有进行对齐! 崩了好多次才发现这个问题, 当场泪崩.
- SIMD指令的优化, 我写了两个版本, 这两个版本的思路区别不大, 只不过第一个是分批装载为m256数据, 第二个是一次性装载为m256数据, 结果第二个实现了优化目标, 第一个却直接来了一个超级加倍优化了-2倍! 现在还有点懵究竟什么导致了这个区别.
- 处理openMP指令的时候, 一开始确实忘了数据竞争的问题, 结果发现输出极其不稳定, 有时候循环个七八次挂了, 有时候一进去就挂了(那时候我还以为不到0.1s就算完了,高兴得起飞-.-)
- 嗷对了, 现在循环要167次才会结束, 之前只需要166次, 但是结果却是完全一致的, 让我百思不得其解



# 参考文献

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- 《并行程序设计导论》机械工业出版社华章公司 (2012)
- [OpenMP Reference Guide](https://www.openmp.org/resources/refguides/)
- [博客-浅谈内存对齐](https://murphypei.github.io/blog/2020/04/memory-align)
- [How is a vector's data aligned?](https://stackoverflow.com/questions/8456236/how-is-a-vectors-data-aligned)
- [AVX指令集加速矩阵乘法](https://www.iiyk.site/archives/211)
