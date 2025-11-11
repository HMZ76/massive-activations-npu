from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
from tqdm import tqdm
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "merged_model", torch_dtype="auto", device_map="auto"
)
with open("./val/classname.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

data_list = {}
# 遍历val目录下的类别文件夹
base_dir = "./val"
for cls_dir in os.listdir(base_dir):
    # 拼接类别文件夹完整路径
    cls_path = os.path.join(base_dir, cls_dir)
    # 跳过文件（只处理文件夹），同时跳过.txt文件
    if not os.path.isdir(cls_path) or cls_dir.endswith('.txt'):
        continue
    # 遍历类别文件夹下的所有图像
    for img_name in os.listdir(cls_path):
        # 拼接图像完整路径（绝对路径）
        img_path = os.path.abspath(os.path.join(cls_path, img_name))
        data_list[img_path] = class_names[int(cls_dir)]
k = list(data_list.keys())[0:100]
data_list = {i: data_list[i] for i in k}

# default processor
processor = AutoProcessor.from_pretrained("merged_model")
right_num = 0
num = 0
for img_path, label in tqdm(data_list.items()):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                },
                {"type": "text", "text": "这是什么种类的垃圾？"},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("npu")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    if output_text[0] == label:
        right_num += 1 
    num +=1
print(right_num/num)