本项目构建了一个主动学习（Active Learning）框架，实现了多个主动学习策略，包括：基于熵（Entropy）、基于最大梯度改变（ECG）、基于聚类（密度峰值聚类）、基于数据增强方式的融合方法等。

## 概要
主动学习方法一般可以分为两部分：学习引擎和选择引擎。学习引擎维护一个基准分类器，并使用监督学习算法对系统提供的已标注样例进行学习从而使该分类器的性能提高，而选择引擎负责运行样例选择算法选择一个未标注的样例并将其交由人类专家进行标注，再将标注后的样例加入到已标注样例集中。学习引擎和选择引擎交替工作，经过多次循环，基准分类器的性能逐渐提高，当满足预设条件时，过程终止。

本项目实现了多种主动学习选择策略（选择引擎），包括：基于熵（Entropy）、基于最大梯度改变（ECG）、基于聚类（密度峰值聚类）、基于数据增强方式的融合方法等，使用CNN模型作为基学习器（学习引擎）构建连续学习框架。

![框架图](https://github.com/Boreaso/ALFramework/raw/master/images/framework.png)

## 项目结构
* models_keras: 使用keras构建模型以及各种基本信息的获取（梯度，深度特征、熵值等）
* models_tf: 使用纯tensorflow接口构建模型
* outputs：数据输出目录
* params：框架参数定义文件（JSON）
* strategies：实现各种主动学习策略（基于熵（Entropy）、基于最大梯度改变（ECG）、基于聚类（密度峰值聚类）、基于数据增强方式的融合方法等）
* utils：包含数据格式转换、音频特征提取、数据存储、结果展示等实现
* framework.py：项目入口，实现了主要项目流程
* hparams.py：项目参数

## 参数说明
    {
      "sub_dir": "urbansound8k",     # 具体任务目录
      "metric_baseline": 0.628606,   # 初始模型准确率基线，每次迭代保存最优的模型参数
      "framework_type": "random",    # 指定不同的主动学习选择引擎
      "num_total": 100000,           # 迭代所用样本总数
      "hist_sel_mode": "certain",    # 使用历史标注样本模式（使用全部|部分|完全不使用）
      "sel_thresholds": [10000],     # 对历史样本进行采样的阈值
      "max_round": 60,               # 增量过程最大迭代次数
      "num_epochs": 50,              # 模型训练迭代纪元数
      "batch_size": 64,              # 批大小
      "decay_rate": 0.001,           # 伪标记阈值衰减系数
      "decay_threshold": 0.046,      # 初始伪标记阈值
      "labeled_percent": 0.1,        # 初始标记数据比例
      "learning_rate": 0.0005,       # 模型学习率
      "load_pretrained": true,       # 是否使用预训练参数
      "num_classes": 10,             # 分类任务类别数量
      "num_select_per_round": 200,   # 每个增量迭代主动选择样本数量
      "pre_train": false,            # 是否进行初始训练阶段
      "random_seed": 1,              # 随机种子
      "test_percent": 0.2,           # 测试集比例
      "valid_percent": 0             # 验证集比例
    }

## 使用
把设置号的配置文件置于params目录下，进入项目顶级目录执行如下代码开始迭代训练过程：

    python framework.py --param_file='params/***.json' 

## 结果
通过对增量训练过程中各个主动学习策略进行性能对比，结果如下：
 
![运行结果](https://github.com/Boreaso/ALFramework/raw/master/images/strategy_result.png)
