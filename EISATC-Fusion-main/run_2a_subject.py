import os
import torch
import torch.nn as nn

from dataLoad.preprocess import get_data
from model import My_Model
from functions import train_with_cross_validate

def main():
    # 1. 路径和基本设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 这里改成你在 Linux 上的实际路径，注意最后有 '/'
    # 用当前脚本所在目录推算 dataLoad/BCICIV_2a 的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "dataLoad", "BCICIV_2a") + "/"

    subject = 1      # 先从被试 1 开始做，后面可以 for 循环跑 1~9
    n_classes = 4    # 2a 是四分类

    # 2. 读取并预处理数据（自动完成裁剪 & 标准化）
    X_train, y_train, X_test, y_test, X_train_trans, y_train_trans = \
        get_data(data_path, subject=subject, LOSO=False,
                 data_model='one_session', data_type='2a')

    print("X_train shape:", X_train.shape)  # 理想情况: (288, 22, 1000)
    print("y_train shape:", y_train.shape)

    # 3. 定义模型
    model = My_Model(eeg_chans=22, samples=1000, n_classes=n_classes, device=device).to(device)

    # 4. 定义 loss 和保存路径
    criterion = nn.CrossEntropyLoss()
    save_dir = "./Saved_files/BCIC_2a/subject-dependent/MyModel/"
    os.makedirs(save_dir, exist_ok=True)

    # 5. 快速冒烟测试配置（几分钟就能跑完，确认不报错）
    train_with_cross_validate(
        model_name      = "MyModel",
        subject         = subject,
        frist_epochs    = 8,      # 先跑8轮就够看是否正常
        eary_stop_epoch = 3,      # 早停耐心值设小
        second_epochs   = 5,      # 第二阶段只跑5轮
        kfolds          = 2,      # 只用2折
        batch_size      = 32,
        device          = device,
        X_train         = X_train,
        Y_train         = y_train,
        model           = model,
        losser          = criterion,
        model_savePath  = save_dir,
        n_calsses       = n_classes
    )

if __name__ == "__main__":
    main()