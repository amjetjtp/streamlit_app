import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import PIL.Image

@st.cache(allow_output_mutation=True)
def load_model():
    class DogCatResNet18(nn.Module):
        def __init__(self):
            super(DogCatResNet18, self).__init__()
            self.feature = models.resnet18(pretrained=False)  # ここでpretrainedはFalseにする
            self.fc = nn.Linear(1000, 2)

        def forward(self, x):
            h = self.feature(x)
            h = self.fc(h)
            return h

    model = DogCatResNet18()

    # state_dictの読み込み
    state_dict = torch.load('dog_cat_ResNet18_01.pt')

    # 'feature.feature.'のプレフィックスを取り除く
    new_state_dict = {k.replace('feature.feature.', 'feature.'): v for k, v in state_dict.items()}

    # 修正したstate_dictをモデルに読み込む
    model.load_state_dict(new_state_dict)

    model.eval()  # 評価モードにセット
    return model

model = load_model()

st.title("犬と猫の分類アプリ")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_column_width=True)

# 画像をモデルが受け入れる形式に変換します
# ... モデルに合わせた前処理
transform_input = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if uploaded_file is not None: # 画像がアップロードされたかどうか確認
    image = PIL.Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_column_width=True)
    
    image_tensor = transform_input(image).unsqueeze(0)  # 画像の前処理とバッチの次元追加
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = outputs.max(1) # max(1)でクラススコアの次元を指定、最大の要素(最大値とそのインデックス)
        # _, で最大値は無視してインデックスのみ取得

    if predicted.item() == 0: # .item()でテンソルが1つの要素だけを含む場合、スカラーとして取り出す
        st.write("これは猫です！")
    else:
        st.write("これは犬です！")
