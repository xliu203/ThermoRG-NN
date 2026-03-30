import os
import sys
import subprocess
from kaggle_secrets import UserSecretsClient

print("🚀 [1/4] 环境初始化与代码拉取...")
%cd /kaggle/working
!rm -rf ThermoRG-NN

# 提取密钥并克隆代码
user_secrets = UserSecretsClient()
gh_token = user_secrets.get_secret("GH_TOKEN")
repo_url = f"https://{gh_token}@github.com/xliu203/ThermoRG-NN.git"
!git clone {repo_url}
%cd /kaggle/working/ThermoRG-NN

print("\n🔧 [2/4] 挂载 ThermoRG 物理引擎...")
!pip install -e .
sys.path.insert(0, '/kaggle/working/ThermoRG-NN/src')
sys.path.insert(0, '/kaggle/working/ThermoRG-NN')

# ✅ 修正架构名称（与 constants.py 中 ALL_SPECS 保持一致）
architectures = [
    "ThermoNet-3", "ThermoNet-5", "ThermoNet-7", "ThermoNet-9",
    "ThermoBot-3", "ThermoBot-5", "ThermoBot-7", "ThermoBot-9",
    "ReLUFurnace-3", "ReLUFurnace-5", "ReLUFurnace-7", "ReLUFurnace-9",
    "ResNet-18-CIFAR", "VGG-11-CIFAR", "DenseNet-40-CIFAR"
]
for arch in architectures:
    os.makedirs(f"experiments/lift_test/results/{arch}", exist_ok=True)

print("\n🔥 [3/4] 启动智能断点续传炼丹炉...")
# 配置 Git 身份
!git config --global user.email "xliu203@asu.edu"
!git config --global user.name "Leo Liu"
!git remote set-url origin {repo_url}

try:
    from experiments.lift_test import train
    train.train_phase_a()
    print("\n🎉 全部架构训练圆满结束！")
except Exception as e:
    print(f"\n🚨 训练中途触发异常: {e}")
    print("⚠️ 别慌！已经触发容灾机制，准备抢救已完成的数据...")
finally:
    print("\n📦 [4/4] 无论成败，执行终极抢救性数据推送...")
    # ✅ 修复：用引号包裹 glob pattern，确保 git add 能正确匹配
    commands = [
        'git add "experiments/lift_test/results/*/*.csv"',
        'git add "experiments/lift_test/results/*/*.json"',
        'git commit -m "🚀 Data: Incremental save from Kaggle via Auto-Resume Script"',
        'git push -u origin main'
    ]
    for cmd in commands:
        # ✅ 修复：result 缩进进 for 循环
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="/kaggle/working/ThermoRG-NN")
        if result.returncode != 0 and "nothing to commit" not in result.stdout and "nothing to commit" not in result.stderr:
            print(f"❌ 推送警告: {result.stderr.replace(gh_token, '***')}")
        else:
            print(f"✅ 执行成功: {cmd.split()[1]}")

    print("\n🏆 抢救性闭环完成！云端机器即将关机。")
