# 相机目标跟踪与位移实验

包含相机标定、模板识别、实时目标跟踪、视频录制和 GNSS 融合分析程序。

## 在另一台 Windows 电脑运行

先安装 Git 和 Python（建议先使用 Python 3.10，与现有相机运行记录一致），连接相机并安装匹配的 Basler pylon 驱动。
在 PowerShell 中执行：

```powershell
git clone https://github.com/d44935143-afk/xiangjijiance.git
cd xiangjijiance
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe test_main --help
.\.venv\Scripts\python.exe test_main
```

私有仓库需要先登录有访问权限的 GitHub 账号。依赖清单根据源码导入整理，尚未在新电脑验证；如已有可用实验环境，优先使用相同的 Python 和依赖版本。

启动前检查 `camera_config/parameters.json` 中的曝光、增益与帧率，详见 `camera_config/README.md`。更换相机、镜头或实验布置后，需要重新标定并初始化目标，检查 `biaoding/camera_intrinsics.json` 和 `targets_state.json` 是否适用于当前实验。

## 常用入口

- `test_main`：相机监测与目标跟踪（文件没有 `.py` 后缀，通过 Python 执行）。
- `record_video.py`：视频录制，可运行 `python record_video.py --help` 查看参数。
- `biaoding/`：相机标定。
- `buchang/`：GNSS 与相机数据分析；运行前检查脚本内输入路径，另行复制所需实验数据。

## 两台电脑同步

每次开始工作前，在项目目录执行：

```powershell
git pull --ff-only
```

修改代码并确认无误后：

```powershell
git status
git add <本次修改的文件路径>
git commit -m "说明本次修改"
git push origin main
```

首次提交前设置自己的 `git config user.name` 和 `git config user.email`。尽量在切换电脑前提交并推送；若拉取提示有未提交修改或分支分叉，先保留修改并处理冲突。

新生成的数据、录像、分析输出、Python 缓存及本机软件由 `.gitignore` 排除。已有历史版本仍包含部分旧数据和缓存；Git 忽略规则不会移除历史文件。需要在两台电脑共享的实验原始数据请单独复制。
