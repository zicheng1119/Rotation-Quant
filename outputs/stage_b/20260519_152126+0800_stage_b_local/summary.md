# Stage B B2 Local Summary

## Linear

| method_key | method | bits | relative_mse | cosine | sqnr_db |
| --- | --- | --- | --- | --- | --- |
| direct_absmax_w4a4 | direct_absmax | W4A4 | 0.038241 | 0.98154 | 15.676126 |
| rot_absmax_w4a4 | rot_absmax | W4A4 | 0.018447 | 0.990991 | 18.778821 |
| rot_lm_w2a4 | rot_lm | W2A4 | 0.09045 | 0.955589 | 11.323754 |
| rot_lm_w3a3 | rot_lm | W3A3 | 0.047051 | 0.976825 | 14.417466 |
| rot_lm_w3a4 | rot_lm | W3A4 | 0.030263 | 0.984977 | 16.423043 |
| rot_lm_w4a3 | rot_lm | W4A3 | 0.029762 | 0.98536 | 16.50877 |
| rot_lm_w4a4 | rot_lm | W4A4 | 0.012809 | 0.993631 | 20.277432 |

## FFN

| method_key | method | bits | relative_mse | cosine | sqnr_db |
| --- | --- | --- | --- | --- | --- |
| ffn_direct_absmax_w4a4 | direct_absmax | W4A4 | 0.101128 | 0.952553 | 10.588065 |
| ffn_fp16 | fp16 | FP16 | 0.0 | 1.000003 | 96.773965 |
| ffn_rot_absmax_w4a4 | rot_absmax | W4A4 | 0.052885 | 0.974711 | 13.103797 |
| ffn_rot_lm_w3a3 | rot_lm | W3A3 | 0.134416 | 0.933755 | 8.735288 |
| ffn_rot_lm_w3a4 | rot_lm | W3A4 | 0.084436 | 0.95783 | 10.832275 |
| ffn_rot_lm_w4a3 | rot_lm | W4A3 | 0.087894 | 0.957444 | 10.577874 |
| ffn_rot_lm_w4a4 | rot_lm | W4A4 | 0.036656 | 0.981933 | 14.461957 |
