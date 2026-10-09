from src.quantization.awq_quantizer import quantize_model


if __name__ == "__main__":
    quantize_model("configs/config.yaml")