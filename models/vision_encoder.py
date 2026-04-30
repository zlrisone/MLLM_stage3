import torch
import torch.nn as nn
from transformers import SiglipVisionModel, SiglipImageProcessor
from typing import Optional, Dict, Any
from PIL import Image

class SigLIPVisionEncoder(nn.Module):
    """
    SigLIP2视觉编码器包装器
    输出768维特征，经过197个patch (14x14 + 1 CLS)
    """

    def __init__(self, model_name: str = "google/siglip2-base-patch16-224",
                 freeze: bool = True):
        """
        初始化视觉编码器

        Args:
            model_name: SigLIP2模型名称
            freeze: 是否冻结权重
        """
        super().__init__()

        # 加载预训练模型
        self.vision_model = SiglipVisionModel.from_pretrained(model_name,trust_remote_code=True)
        self.processor = SiglipImageProcessor.from_pretrained(model_name)
        
        # 冻结参数
        if freeze:
            for param in self.vision_model.parameters():
                param.requires_grad = False

        self.freeze = freeze

        # 获取模型配置
        self.config = self.vision_model.config
        self.hidden_size = self.config.hidden_size  # 768
        self.num_patches = (self.config.image_size // self.config.patch_size) ** 2  # 196
        self.output_dim = self.hidden_size

        print(f"SigLIP Vision Encoder loaded: {model_name}")
        print(f"Hidden size: {self.hidden_size}, Patches: {self.num_patches}")
        print(f"Freeze: {freeze}")
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            pixel_values: 图像像素值 (B, 3, H, W)

        Returns:
            视觉特征 (B, 197, 768) - 196个patch + 1个CLS token
        """
        # SigLIP2输出包含last_hidden_state
        outputs = self.vision_model(pixel_values=pixel_values, output_hidden_states=True)

        # patch_features = outputs.last_hidden_state      # [B, N, C]
        # pooler_output = outputs.pooler_output # 全局特征 [B, 768]
        # print(pooler_output.shape)
        patch_features_lm = outputs.hidden_states[-2] 
        # 返回last_hidden_state: (B, 196, 768)
        return patch_features_lm

    def preprocess(self, images):
        """
        预处理图像

        Args:
            images: 输入图像 (PIL Image, tensor等)

        Returns:
            处理后的tensor (B, 3, H, W)
        """
        # 使用SigLIP的processor进行预处理
        if isinstance(images, list):
            processed = self.processor(images, return_tensors="pt")
        else:
            processed = self.processor(images, return_tensors="pt")

        return processed['pixel_values']
        
if __name__ =="__main__":
    model = SigLIPVisionEncoder()
    image = Image.open("/Users/lris/Desktop/HIT/LLM/MMLLMs/LLaVA_重新实现/stage2/Electric Shock - EP.PNG").convert("RGB")
    pixel_values = model.preprocess(image)
    print(pixel_values.shape)
    outputs = model(pixel_values)
    print(outputs.shape)