# app_qwen2vl_gradio.py
import os
import torch
import gradio as gr
from modelscope import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ========== 模型与处理器 ==========
# 可改为 bfloat16 + flash_attention_2（需安装 flash-attn 且显卡支持）
USE_FLASH_ATTENTION_2 = False

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

if USE_FLASH_ATTENTION_2:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
else:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )

processor = AutoProcessor.from_pretrained(MODEL_ID)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ========== 推理函数 ==========
def answer_image_question(image, question, max_new_tokens=128, temperature=0.7, top_p=0.9, do_sample=True):
    """
    image: PIL.Image 或 None（来自 gr.Image(type="pil")）
    question: 用户输入问题
    """
    if image is None:
        return "请先上传一张图片。"
    if not question or not question.strip():
        question = "Describe this image."

    # 构建多模态消息（保持与你示例一致的结构）
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},           # 这里直接传 PIL.Image
                {"type": "text", "text": question.strip()},
            ],
        }
    ]

    # 模板 + 视觉预处理
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    # 组装 Batch 输入
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # 将输入移动到设备（与模型保持一致）
    try:
        inputs = inputs.to(DEVICE)
    except Exception:
        # 当模型被分布到多设备时（device_map="auto"），inputs.to("cuda") 也通常可行；
        # 若失败，回退保持原状，依然可推理（Accelerate会处理）。
        pass

    # 生成
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_p=float(top_p),
        )

    # 截去输入部分，仅保留新增tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # 解码
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


# ========== Gradio UI ==========
def build_demo():
    with gr.Blocks(title="Qwen2-VL 韩语图文问答\n") as demo:
        gr.Markdown(
            """
            <div style="text-align:center; line-height:1.35; margin: 6px 0 14px;">
            <div style="font-size:2.1em; font-weight:700;">Qwen2-VL 韩语图文问答</div>
            <div style="font-size:1.15em;">
                上传图片 + 输入问题，点击 <b>回答</b> 即可。<br/>
                模型：<code>Qwen/Qwen2-VL-2B-Instruct</code>
            </div>
            </div>
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="pil", label="上传图片（支持拖拽）")
            with gr.Column(scale=1):
                q = gr.Textbox(label="问题", value="Describe this image.", lines=3, placeholder="请输入你的问题")
                max_new_tokens = gr.Slider(16, 1024, value=128, step=8, label="max_new_tokens")
                temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p")
                do_sample = gr.Checkbox(value=True, label="do_sample（采样生成）")
                btn = gr.Button("回答", variant="primary")

        out = gr.Textbox(label="回答", lines=8)

        # 事件绑定
        btn.click(
            fn=answer_image_question,
            inputs=[img, q, max_new_tokens, temperature, top_p, do_sample],
            outputs=[out]
        )

        # 可选：示例（需要联网才能加载URL）
        gr.Examples(
            examples=[
                ["https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg", "Describe this image.", 128, 0.7, 0.9, True],
            ],
            inputs=[img, q, max_new_tokens, temperature, top_p, do_sample],
            label="示例"
        )

        gr.Markdown(
            """
            **注意**
            - 显存不够：可减小 `max_new_tokens`，或在上方代码中设置 `min_pixels/max_pixels` 限制图像分辨率。
            """
        )
    return demo


if __name__ == "__main__":
    demo = build_demo()
    # 若需要公网分享，share=True；如需指定端口，传 server_port=7860
    demo.queue().launch(server_port=7861)
