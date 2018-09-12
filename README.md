## 第三届阿里云安全算法挑战赛 线上第七名队伍 爱拼才会赢@
## A榜单线上0.062  B榜线上0.0476

文档说明:
data_proprecesseing_xx.py 讲原始的csv文件种的api和返回值 按照顺序排序在api_text,reteurn_value_text字段

data_unit.py 在处理过后的csvs文件中提取tfidf特征 统计特征 组合特征 用npz文件保存矩阵，方便训练

其他文件 不同模型读取保存的npz矩阵 进行训练

单模型最高tfidf+count_feature_lgb.py  A榜单0.062
模型融合stacking A榜 0.059
