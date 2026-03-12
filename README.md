# CS336 Practice - Day 1

## 项目简介
这是我第一次系统学习 CS336（Language Modeling from Scratch）的练习仓库。  
当前目标是按作业/课程要求逐步完成基础组件，实现可测试、可复现的代码。

## 当前进度
- 已完成 `tokenizer` 部分的核心实现与桥接。
- 已打通测试适配层，使测试可以直接调用包内实现。

## 已完成内容（Tokenizer）
- 实现了基于 BPE merges 的编码逻辑（`encode`）。
- 实现了解码逻辑（`decode`）。
- 实现了流式编码接口（`encode_iterable`）。
- 支持 special tokens（包括重叠 special token 的处理）。
- 在适配层中将 `get_tokenizer` 桥接到包内实现，减少重复代码。

## 测试结果
已通过 tokenizer 相关测试：
- `tests/test_tokenizer.py` 通过（本地结果：23 passed，2 skipped）。

## 运行方式
在项目根目录执行：

```bash
# 推荐 UTF-8 模式运行测试
python -X utf8 -m pytest tests/test_tokenizer.py -q
```
## 说明
这是我第一次系统学习 CS336，很多实现还在摸索和迭代中。  
如果有理解不准确、实现不够优雅或工程细节处理不当的地方，欢迎各位大佬批评指正，我会认真学习并持续改进。感谢！
