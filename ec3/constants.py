from pathlib import Path
IMAGE_RES = 30
TOKLEN = 30
P_CTX = 64
POSLEN = P_CTX // 2
P_INDIM = TOKLEN + 1 + POSLEN
E_INDIM = 5 + TOKLEN + POSLEN

EC3_ROOT = Path(__file__).parent
CHECKPOINTS_ROOT = EC3_ROOT / "checkpoints"
RECOGNIZER_CHECKPOINT_SAVEPATH = CHECKPOINTS_ROOT / "recognizer_checkpoint.ptx"
