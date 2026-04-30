import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM,AutoTokenizer

from peft import LoraConfig, get_peft_model, PeftModel
from typing import Optional, Dict, Any, List
import torch.nn.functional as F

class QwenDecoder(nn.Module):
    """
    Qwen2.5语言模型解码器包装器
    支持Lora微调和全量微调
    """
    def __init__(self, model_name:str="Qwen/Qwen2.5-3B", freeze:bool=True, use_lora:bool=False, lora_config:Optional[dict]=None):
        """
        初始化Qwen解码器
        Args:
            model_name: Qwen模型名称
            freeze: 是否冻结权重 (全量微调时为False)
            use_lora: 是否使用LoRA
            lora_config: LoRA配置字典
        """
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self.use_lora = use_lora

        # 加载预训练模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map = None,
            trust_remote_code=True,
        )
        
        # 获取模型配置
        self.config = self.model.config
        self.vocab_size = self.config.vocab_size # 151936
        self.hidden_size = self.config.hidden_size # 2048

        # 应用LoRA
        if use_lora:
            self._apply_lora(lora_config)
        
        # 冻结参数
        if freeze and not use_lora:
            for param in self.model.parameters():
                param.requires_grad = False
        
        print(f"Qwen Decoder loaded: {model_name}")
        print(f"Hidden size: {self.hidden_size}, Vocab size: {self.vocab_size}")
        print(f"Freeze: {freeze}, LoRA: {use_lora}")
        print(f"Trainable params: {self.get_trainable_params():,}")

    def _apply_lora(self, lora_config:dict):
        """应用LoRA配置"""
        # 默认LoRA配置
        default_config = {
            'r': 64,
            'lora_alpha': 16,
            'lora_dropout': 0.05,
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            'lora_layers': None  # None表示所有层
        }

        # 合并配置
        if lora_config:
            default_config.update(lora_config)

        # 构建LoRA配置
        lora_config = LoraConfig(
            r=default_config['r'],
            lora_alpha=default_config['lora_alpha'],
            lora_dropout=default_config['lora_dropout'],
            target_modules=default_config['target_modules'],
            bias="none",
            task_type="CAUSAL_LM"
        )

        # 应用LoRA
        self.model = get_peft_model(self.model, lora_config)

        # 如果指定了层数，只在最后N层应用LoRA
        if default_config['lora_layers'] is not None:
            self._apply_lora_to_layers(default_config['lora_layers'])
    def _apply_lora_to_layers(self, num_layers:int):
        """只在最后N层应用LoRA"""
        # 获取模型的所有层
        layers = self.model.base_model.model.layers

        # 计算要应用LoRA的层索引
        total_layers = len(layers)
        start_layer = max(0, total_layers - num_layers)

        print(f"Applying LoRA to layers {start_layer} to {total_layers-1}")

        # 冻结前面层的LoRA参数
        for i in range(start_layer):
            for name, param in layers[i].named_parameters():
                if 'lora_' in name:
                    param.requires_grad = False
    def forward(self, input_ids:torch.Tensor, attention_mask: Optional[torch.tensor] = None, labels : Optional[torch.tensor]=None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: 输入token ids (B, seq_len)
            attention_mask: 注意力掩码 (B, seq_len)
            labels: 标签 (用于计算loss) (B, seq_len)
            **kwargs: 其他参数

        Returns:
            模型输出字典
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        return {
            'logits': outputs.logits,  # (B, seq_len, vocab_size)
            'loss': outputs.loss if hasattr(outputs, 'loss') else None
        }
    def get_trainable_params(self) -> int:
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def load_adapter(self, adapter_path: str):
        """加载LoRA适配器"""
        if self.use_lora:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

    def merge_and_unload(self):
        """合并LoRA权重并卸载LoRA模块"""
        if self.use_lora:
            self.model = self.model.merge_and_unload()
            self.use_lora = False
            
if __name__=="__main__":
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = QwenDecoder(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # print(tokenizer.additional_special_tokens)
    # print(tokenizer.additional_special_tokens_ids)
    # prepare the model input
    
    prompt = "Describe this photo.<|vision_start|><|image_pad|><|vision_end|>"
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print(text)
    text = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Describe this photo.<|vision_start|><|image_pad|><|vision_end|><|im_end|>
<|im_start|>assistant\n"""
    print(text)
    model_inputs = tokenizer([text], return_tensors="pt")
    text_embeddings = model.model.get_input_embeddings()(model_inputs.input_ids)
    print(text_embeddings.shape)

    projected_features = torch.ones(1,196,896)
    print(projected_features.shape)

    input_embeddings = torch.cat([text_embeddings,projected_features], dim=1)
    print(input_embeddings.shape)
    generated_ids = model.model.generate(
        inputs_embeds=input_embeddings,
        max_new_tokens=512
    )
    print(generated_ids.shape)

    # generated_ids = [
    #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    # ]
    inputs = tokenizer.batch_decode(model_inputs.input_ids, skip_special_tokens=False)[0]
    print('-----------')
    print(inputs)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print('-----------')
    
    print(response)
#     prompt = """<|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# Describe this photo.<|vision_start|><|image_pad|><|vision_end|><|im_end|>
# <|im_start|>assistant"""
#     full = """<|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# Describe this photo.<|vision_start|><|image_pad|><|vision_end|><|im_end|>
# <|im_start|>assistant
# This is a cat."""
#     prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
#     model_inputs = tokenizer(full, return_tensors="pt", padding="longest", max_length=128, truncation=True)
#     input_ids = model_inputs["input_ids"]
#     attention_mask = model_inputs["attention_mask"]

#     labels = input_ids.clone()
#     prompt_len = prompt_ids.shape[1]
#     labels[:, :prompt_len] = -100

#     outputs = model(input_ids,attention_mask,labels)
#     print(outputs['loss'].item())
    