# [261017] 02_Computer-Vision

모폴로지 연산을 적용하여 최종 농구공 Contour 선택 및 출력

| 파일 | 내용 |
|---|---|
| `basketball_contour.py` | LAB + CLAHE → Binary Mask → Opening x2 / Closing x3 → Contour 검출 → 가장 큰 Contour(농구공) 선택 → 최소 외접원·무게중심 표시 |
| `basketball_result.png` | 실행 결과 |

```bash
python3 261017/basketball_contour.py
```
