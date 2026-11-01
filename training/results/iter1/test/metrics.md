# iter1 (test split)

mAP50: **0.522** · mAP50-95: **0.366**

| class | precision | recall | AP50 | AP50-95 |
|---|---|---|---|---|
| ball | 0.633 | 0.706 | 0.670 | 0.445 |
| can | 0.645 | 0.449 | 0.478 | 0.391 |
| box | 0.614 | 0.375 | 0.426 | 0.287 |
| bottle | 0.763 | 0.446 | 0.514 | 0.338 |

Main errors (true → predicted; 'background' = missed / false positive):
- bottle → predicted background: 148
- box → predicted background: 106
- can → predicted background: 81
- background → predicted ball: 63
- background → predicted box: 60
