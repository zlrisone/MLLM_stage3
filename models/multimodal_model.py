import torch
import torch.nn as nn
import torch.nn.functional as F

from models.llm import QwenDecoder
from models.projector import LinearProjector,MLPProjector,DeepMLPProjector
from models.vision_encoder import SigLIPVisionEncoder

from typing import Optional, Dict, Any, List

class MultimodalModel(nn.Module):
    """
    端到端多模态模型
    Vision Encoder + Projector + LLM Decoder
    """

    def __init__(self, config: dict):
        """
        初始化多模态模型

        Args:
            config: 完整配置字典
        """
        super().__init__()

        self.config = config
        model_cfg = config["model"]
        # 构建各个组件
        # 1) LLM
        self.llm_decoder = QwenDecoder(
            model_name=model_cfg["llm"]["model_name"],
            freeze=model_cfg["llm"]["freeze"],
            use_lora=model_cfg["llm"]["use_lora"],
        )

        # 2) Vision Encoder
        self.vision_encoder = SigLIPVisionEncoder(
            model_name=model_cfg["vision_encoder"]["model_name"],
            freeze=model_cfg["vision_encoder"]["freeze"],
        )

        # 3) Projector
        projector_type = model_cfg["projector"]["type"].lower()
        if projector_type == "linear":
            self.projector = LinearProjector(
                input_dim=model_cfg["LinearProjector"]["input_dim"],
                output_dim=self.llm_decoder.hidden_size,
            )

        elif projector_type == "mlp":
            self.projector = MLPProjector(
                input_dim=model_cfg["MLPProjector"]["input_dim"],
                hidden_dim=model_cfg["MLPProjector"]["hidden_dim"],
                output_dim=self.llm_decoder.hidden_size,
                activation=model_cfg["MLPProjector"]["activation"],
                dropout=model_cfg["MLPProjector"]["dropout"],
            )

        elif projector_type == "deepmlp":
            self.projector = DeepMLPProjector(
                input_dim=model_cfg["DeepMLPProjector"]["input_dim"],
                hidden_dim=model_cfg["DeepMLPProjector"]["hidden_dim"],
                output_dim=self.llm_decoder.hidden_size,
                dropout=model_cfg["DeepMLPProjector"]["dropout"],
            )

        else:
            raise ValueError(f"Unsupported projector type: {projector_type}")
        
        self.image_pad_token_id = model_cfg["llm"]["image_pad_id"]

    def _combine_vision_text_embeddings(self, vision_embeds: torch.Tensor,
                                       text_embeds: torch.Tensor,
                                       input_ids: torch.Tensor) -> torch.Tensor:
        """
        组合视觉和文本嵌入

        Args:
            vision_embeds: (B, N, H)
            text_embeds:   (B, L, H)
            input_ids:     (B, L)

        Returns:
            combined_embeds: (B, L - num_image_tokens + N * num_image_tokens, H)

        当前实现假设每个样本只有 1 个 <|image_pad|>。
        """
        B, L, H = text_embeds.shape
        Bv, N, Hv = vision_embeds.shape

        if B != Bv:
            raise ValueError(f"Batch mismatch: text={B}, vision={Bv}")
        if H != Hv:
            raise ValueError(f"Hidden size mismatch: text={H}, vision={Hv}")
        
        combined_list = []

        for b in range(B):
            cur_input_ids = input_ids[b]         # [L]
            cur_text_embeds = text_embeds[b]     # [L, H]
            cur_vision_embeds = vision_embeds[b] # [N, H]

            image_pos = (cur_input_ids == self.image_pad_token_id).nonzero(as_tuple=False).squeeze(-1)

            if image_pos.numel() == 0:
                raise ValueError(f"Sample {b} has no <|image_pad|> token.")
            if image_pos.numel() > 1:
                raise ValueError(
                    f"Sample {b} has {image_pos.numel()} <|image_pad|> tokens. "
                    "Current implementation only supports one image token per sample."
                )

            pos = image_pos.item()

            # 保留 image_pad 之前的文本
            left = cur_text_embeds[:pos]          # [pos, H]

            # 跳过 image_pad 本身，用 vision_embeds 替换
            right = cur_text_embeds[pos + 1:]     # [L-pos-1, H]

            combined = torch.cat([left, cur_vision_embeds, right], dim=0)  # [L-1+N, H]
            combined_list.append(combined)

        # 当前每个样本的 N 一样，所以长度一致，可以 stack
        combined_embeds = torch.stack(combined_list, dim=0)  # [B, L-1+N, H]
        return combined_embeds

    def _expand_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        num_vision_tokens: int,
    ) -> torch.Tensor:
        """
        将 attention_mask 中的 <|image_pad|> 位置扩展成 num_vision_tokens 个 1
        """
        B, L = attention_mask.shape
        expanded_list = []

        for b in range(B):
            cur_mask = attention_mask[b]
            cur_input_ids = input_ids[b]

            image_pos = (cur_input_ids == self.image_pad_token_id).nonzero(as_tuple=False).squeeze(-1)

            if image_pos.numel() != 1:
                raise ValueError(f"Sample {b} must contain exactly one <|image_pad|> token.")

            pos = image_pos.item()

            left = cur_mask[:pos]
            right = cur_mask[pos + 1:]

            vision_mask = torch.ones(
                num_vision_tokens,
                dtype=cur_mask.dtype,
                device=cur_mask.device
            )

            expanded = torch.cat([left, vision_mask, right], dim=0)
            expanded_list.append(expanded)

        return torch.stack(expanded_list, dim=0)
    def _expand_labels(
        self,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
        num_vision_tokens: int,
    ) -> torch.Tensor:
        """
        将 labels 中 <|image_pad|> 对应位置替换为 num_vision_tokens 个 -100
        """
        B, L = labels.shape
        expanded_list = []

        for b in range(B):
            cur_labels = labels[b]
            cur_input_ids = input_ids[b]

            image_pos = (cur_input_ids == self.image_pad_token_id).nonzero(as_tuple=False).squeeze(-1)

            if image_pos.numel() != 1:
                raise ValueError(f"Sample {b} must contain exactly one <|image_pad|> token.")

            pos = image_pos.item()

            left = cur_labels[:pos]
            right = cur_labels[pos + 1:]

            vision_labels = torch.full(
                (num_vision_tokens,),
                -100,
                dtype=cur_labels.dtype,
                device=cur_labels.device
            )

            expanded = torch.cat([left, vision_labels, right], dim=0)
            expanded_list.append(expanded)

        return torch.stack(expanded_list, dim=0)
    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            images: 图像tensor (B, 3, H, W)
            input_ids: 文本token ids (B, seq_len)
            attention_mask: 注意力掩码 (B, seq_len)
            labels: 标签 (B, seq_len)
            **kwargs: 其他参数

        Returns:
            模型输出字典
        """
        # 1. 视觉编码
        vision_features = self.vision_encoder(pixel_values) # (B, N, D)
        
        # 2. 投影到语义空间
        projected_features = self.projector(vision_features)    # (B, N, H)
        
        # 3. 准备输入序列
        text_embeddings = self.llm_decoder.model.get_input_embeddings()(input_ids)  # (B, seq_len, H)
        projected_features = projected_features.to(
            device=text_embeddings.device,
            dtype=text_embeddings.dtype
        )
        # 找到图像token的位置并替换
        combined_embeddings = self._combine_vision_text_embeddings(
            projected_features, text_embeddings, input_ids
        ) # [B, L-1+N, H]
        if attention_mask is not None:
            attention_mask = self._expand_attention_mask(
                attention_mask, input_ids, projected_features.size(1)
            )
        if labels is not None:
            labels = self._expand_labels(
                labels, input_ids, projected_features.size(1)
            )

        # 4. LLM前向传播
        outputs = self.llm_decoder.model(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        # 准备返回结果
        result = {
            'logits': outputs.logits,
            'loss': outputs.loss if hasattr(outputs, 'loss') else None,
            'vision_features': vision_features,
            'projected_features': projected_features
        }
        return result

    def generate(self, pixel_values: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                max_new_tokens: int = 100, return_prompt_length: bool = False, **kwargs) -> torch.Tensor:
        """
        生成文本

        Args:
            images: 图像tensor (B, 3, H, W)
            input_ids: 输入token ids (B, seq_len)
            attention_mask: 注意力掩码 (B, seq_len)
            max_new_tokens: 最大生成token数
            **kwargs: 生成参数

        Returns:
            生成的token序列
        """
        # 1. 视觉编码
        vision_features = self.vision_encoder(pixel_values)          # [B, N, Dv]

        # 2. 投影到 LLM hidden space
        projected_features = self.projector(vision_features)         # [B, N, H]

        # 3. 文本 embedding
        text_embeddings = self.llm_decoder.model.get_input_embeddings()(input_ids)  # [B, L, H]
        projected_features = projected_features.to(
            device=text_embeddings.device,
            dtype=text_embeddings.dtype
        )
        # 4. 用视觉 token 替换 <|image_pad|>
        combined_embeddings = self._combine_vision_text_embeddings(
            projected_features, text_embeddings, input_ids
        )  # [B, L-1+N, H]

        # 5. attention_mask 同步扩展
        if attention_mask is not None:
            combined_attention_mask = self._expand_attention_mask(
                attention_mask=attention_mask,
                input_ids=input_ids,
                num_vision_tokens=projected_features.size(1),
            )
        else:
            combined_attention_mask = torch.ones(
                combined_embeddings.size(0),
                combined_embeddings.size(1),
                dtype=torch.long,
                device=combined_embeddings.device,
            )
        prompt_length = combined_attention_mask.size(1)
        # 6. generate
        with torch.no_grad():
            generated_ids = self.llm_decoder.model.generate(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.llm_decoder.model.config.pad_token_id,
                eos_token_id=self.llm_decoder.model.config.eos_token_id,
                **kwargs
            )

        if return_prompt_length:
            return generated_ids, prompt_length
        return generated_ids

def create_multimodal_model(config: dict) -> MultimodalModel:
    """
    创建多模态模型

    Args:
        config: 配置字典

    Returns:
        多模态模型实例
    """
    return MultimodalModel(config)