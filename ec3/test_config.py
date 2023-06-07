from config import ModelConfig

def test_creating_default_instance():
    model_config = ModelConfig()
    
def test_derived_properties():

    # Act
    model_config = ModelConfig(p_ctx=10, toklen=15)

    # Assert
    assert model_config.poslen == 5
    
    assert model_config.p_indim == 21 #15 + 1 + 5
    
    assert model_config.e_indim == 25 #5 + 15 + 5