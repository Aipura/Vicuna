# Aipura
大模型训练脚本，基于 FastChat 框架，对**数据模块**和**参数模块**进行了拓展，除了 [FastChat](https://github.com/lm-sys/FastChat) 官方的 Vicuna 的数据格式，额外支持 Alpaca 等数据格式的指令微调。

## 结构
----data  数据

----playground  DeepSpeed 配置等

----scripts  训练脚本

----src  数据模块/参数模块

----train.py  全量训练

----train_lora.py  (Q)LoRA 训练


## 快速开始
### 克隆本仓库
```bash
git clone https://github.com/Aipura/Aipura.git
```
### 安装依赖
```bash
cd Aipura
pip3 install -r requirements.txt
```
### 准备数据
示例数据 data/alpaca_data_cleaned_1000.json

### 编辑训练脚本
以 LoRA 为例
```
vim scripts/train_llama2_7b_lora.sh
```
### 开始训练
以 LoRA 为例
```
bash scripts/train_llama2_7b_lora.sh
```
