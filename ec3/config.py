from pydantic import BaseModel, Field, validator

class ModelConfig(BaseModel):
    # global
    dreaming: bool = Field(False, description="Dreaming mode")
    
    # data
    image_res: int = Field(30, description="Image resolution")
    toklen: int = Field(30, description="Token length")
    p_ctx: int = Field(64, description="Context for program")
    poslen: int = Field(32, description="Position length")
    p_indim: int = Field(63, description="Program input dimension")
    e_indim: int = Field(67, description="Encoder input dimension")
    patch_size: int = Field(5, description="Patch size for vision model")
    # Recognizer model
    v_ctx: int = Field(None, description="Context for vision model")
    vision_width: int = Field(256, description="Vision width")
    prog_width: int = Field(256, description="Program width")
    vision_heads: int = Field(8, description="Number of vision heads")
    vision_layers: int = Field(6, description="Number of vision layers")
    prog_heads: int = Field(8, description="Number of program heads")
    prog_layers: int = Field(8, description="Number of program layers")
    embed_dim: int = Field(256, description="Embedding dimension")
    # training
    train_iters: int = Field(100000, description="Number of training iterations")
    learning_rate: float = Field(0.00025, description="Learning rate")
        # model is quite sensitive to learning rate 
        # 0.0005 is a good start, but causes oscillations later
        # 0.00025 is better after 2000 batches of 512.
    weight_decay: float = Field(2.5e-6, description="Weight decay")
    nreplace: int = Field(0, description="Number of replacements")
    batch_size: int = Field(32, description="Batch size")
    
    @validator('v_ctx', pre=True, always=True)
    def set_v_ctx_default(cls, v, values, **kwargs):
        if v is not None:
            return v
        if 'image_res' in values and 'patch_size' in values:
            return int((values['image_res'] / values['patch_size']) ** 2 + 1)
        return v
    
    @property
    def training(self):
        return not self.dreaming
    
    @property
    def mmapno(self):
        return 1 if self.dreaming else 0
    
    @property
    def edsiz(self):
        return self.batch_size * self.e_indim * 4
    
    @property
    def edsiz_allocate_command(self):
        return f"fallocate -l {self.edsiz} editdiff_{self.mmapno}.mmap"
