# code 目录说明

`code/` 用于承载《从零实现到PyTorch：深度学习算法入门与进阶》对应的实战代码。

它不是临时练习目录，而是整套教材的代码主仓。后续 `MNIST`、`CNN`、`ResNet`、`Transformer`、`Diffusion` 等阶段的实验、示例、训练脚本和公共模块，都应逐步沉淀到这里。

## 设计目标

这个目录结构需要同时支持以下几件事：

1. 先快速跑通当前的 `MNIST` 项目
2. 后续自然扩展到 `CNN`、`ResNet`、`Transformer`、`Diffusion`
3. 把教材章节和代码目录建立稳定映射
4. 逐步沉淀公共模块，而不是每一阶段都从头复制代码
5. 兼顾“教学示例代码”和“可重复运行的项目代码”

## 顶层目录规划

```text
code/
├── readme.md
├── configs/
├── docs/
├── notebooks/
├── outputs/
├── scripts/
└── src/
    ├── common/
    └── projects/
```

## 各目录职责

### `configs/`

用于存放训练配置、实验参数和后续不同模型的配置文件。

### `docs/`

用于存放和代码直接相关的小型文档，不替代主教材。

主教材统一放在 `book/`，这里只放代码侧的运行说明、实验记录和补充说明。

### `notebooks/`

用于存放探索性实验、数据观察和临时分析笔记。

### `outputs/`

用于存放训练日志、模型权重、图片结果和评估报告等运行产物。

### `scripts/`

用于存放直接执行的脚本入口。

原则：

1. `scripts/` 负责调用
2. `src/` 负责实现

### `src/common/`

用于存放不同项目都会复用的公共模块。

当前已规划子目录：

1. `data/`
2. `engine/`
3. `metrics/`
4. `utils/`
5. `visualization/`

### `src/projects/`

用于按主题拆分不同阶段的项目代码。

```text
src/projects/
├── mnist/
├── cnn/
├── resnet/
├── transformer/
└── diffusion/
```

## 当前阶段重点：MNIST

当前最先需要使用的是：

```text
src/projects/mnist/
├── baseline/
├── from_scratch/
└── pytorch/
```

### `baseline/`

放当前阶段第一个最小可运行版本，用来先跑通数据、模型、训练和评估的完整流程。

### `from_scratch/`

放从零实现版本，用更底层的方式理解前向传播、损失和参数更新。

### `pytorch/`

放 `PyTorch` 对照版本，把已经理解过的底层概念映射到主流框架。

## 教材与代码映射原则

后续尽量遵循下面的映射方式：

1. `book/` 中的 `MNIST` 卷，对应 `code/src/projects/mnist/`
2. `book/` 中的 `CNN` 卷，对应 `code/src/projects/cnn/`
3. `book/` 中的 `ResNet` 卷，对应 `code/src/projects/resnet/`
4. `book/` 中的 `Transformer` 卷，对应 `code/src/projects/transformer/`
5. `book/` 中的 `Diffusion` 卷，对应 `code/src/projects/diffusion/`

## 建议迭代方式

建议后续按以下顺序推进：

1. 先在 `code/src/projects/mnist/baseline/` 跑通最小版本
2. 再补 `code/src/projects/mnist/from_scratch/`
3. 再整理 `code/src/projects/mnist/pytorch/`
4. 在这个过程中，把公共逻辑逐步抽到 `code/src/common/`
5. 等 `MNIST` 稳定后，再复制这套模式扩展到其他主题
