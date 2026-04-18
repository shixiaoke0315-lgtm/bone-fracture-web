from flask import Flask, request, render_template
import torch
import timm
from torchvision import transforms
from PIL import Image
import json
import os

app = Flask(__name__)

# ======================
# 1. 加载类别名称（关键修改）
# ======================
with open("classes.json", "r") as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)

print("Loaded classes:", class_names)

# ======================
# 2. 设备
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# 3. 加载模型
# ======================
model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=NUM_CLASSES
)

WEIGHT_PATH = "bone_fracture_model.pth"

if not os.path.exists(WEIGHT_PATH):
    raise FileNotFoundError("❌ 模型文件不存在，请检查路径")

model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
model.to(device)
model.eval()

print("✅ Model loaded successfully")

# ======================
# 4. 图像预处理（必须和训练一致）
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ======================
# 5. 首页
# ======================
@app.route('/')
def home():
    return render_template('index.html')

# ======================
# 6. 预测接口（增强版：返回Top-3概率）
# ======================
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="未上传文件")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction="未选择文件")

    try:
        img = Image.open(file).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)

            top3_prob, top3_idx = torch.topk(probs, 3)

        results = []
        for i in range(3):
            label = class_names[top3_idx[0][i].item()]
            prob = top3_prob[0][i].item() * 100
            results.append(f"{label}: {prob:.2f}%")

        return render_template('index.html', prediction=results)

    except Exception as e:
        return render_template('index.html', prediction=f"错误: {str(e)}")

# ======================
# 7. 启动
# ======================
if __name__ == '__main__':
    app.run(debug=True)