import gradio as gr
from ultralytics import YOLO

# 加载你自己训练出来的安全帽模型
model = YOLO('best.pt')

def predict(image, conf_threshold):
    # 进行推理，设置置信度阈值
    results = model(image, conf=conf_threshold)
    # 返回画了框的图片
    return results[0].plot()

# 搭建网页界面
with gr.Blocks(title="安全帽检测系统") as demo:
    gr.Markdown("# 👷 智能工地安全帽佩戴检测系统")
    gr.Markdown("基于 YOLOv8 微调模型。上传工地照片，实时识别人员是否佩戴安全帽。")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="numpy", label="上传图片")
            conf_slider = gr.Slider(minimum=0.1, maximum=0.9, value=0.25, label="置信度阈值 (Confidence Threshold)")
            btn = gr.Button("开始检测", variant="primary")
        with gr.Column():
            img_output = gr.Image(type="numpy", label="检测结果")
            
    btn.click(fn=predict, inputs=[img_input, conf_slider], outputs=img_output)

if __name__ == "__main__":
    demo.launch()