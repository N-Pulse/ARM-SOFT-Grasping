# iter1 (val split)

mAP50: **0.574** · mAP50-95: **0.388**

| class | precision | recall | AP50 | AP50-95 |
|---|---|---|---|---|
| ball | 0.712 | 0.724 | 0.683 | 0.441 |
| can | 0.703 | 0.465 | 0.523 | 0.443 |
| box | 0.629 | 0.487 | 0.500 | 0.317 |
| bottle | 0.825 | 0.489 | 0.588 | 0.352 |

Main errors (true → predicted; 'background' = missed / false positive):
- bottle → predicted background: 126
- can → predicted background: 79
- box → predicted background: 55
- background → predicted ball: 45
- background → predicted box: 42
