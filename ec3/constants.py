from pathlib import Path

EC3_ROOT = Path(__file__).parent
CHECKPOINTS_ROOT = EC3_ROOT / "checkpoints"

image_res = 30
toklen = 30
p_ctx = 96
poslen = 7 # p_ctx // 2
p_indim = toklen + 1 + poslen 
e_indim = 5 + toklen + poslen
