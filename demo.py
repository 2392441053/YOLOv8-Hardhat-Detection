from ultralytics import YOLO

print("正在加载官方预训练模型（第一次运行会自动下载，大概几秒钟）...")
# yolov8n.pt 是 YOLOv8 最小的模型，速度最快
model = YOLO('yolov8n.pt') 

print("模型加载成功！开始识别图片...")
results = model('test.jpg') 

# 把画了框的图片保存下来
results[0].save('result_demo.jpg')
print("识别完成！请去文件夹里查看 result_demo.jpg")