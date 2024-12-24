# Robocup3D Darkblue 球队环境部署教程

## 什么是 Robocup3D

Robocup 是一项国际科学倡议，旨在推动智能机器人的发展的比赛。为了实现这一目标，RoboCup在不同领域提出了几个不同的比赛，Robocup3D 使用虚拟3D机器人来模拟真正的足球比赛。

## 环境依赖构建

Robocup3D 主要由服务器(rcssserver3d)，监视器(Roboviz)，和球队组成，整个环境运行在 Linux 下，因此需要安装 Ubuntu 或其他 Linux 环境。

### 1. 安装 Windows Subsystem For Linux (WSL)

[安装教程](https://blog.csdn.net/wangtcCSDN/article/details/137950545)

### 2. 安装服务器(rcssserver3d)

[rcssserver3d 下载链接](https://software.opensuse.org/download.html?project=science:SimSpark&package=rcssserver3d)

### 3.安装监视器(Roboviz)

Roboviz 需要 Java 运行环境（JRE）才能运行，因此具备跨平台的能力。它可以在 Windows 和 Linux 系统上使用，但建议在两者上都安装 Java 环境。

### Windows

Windows 平台可以下载 [OpenJDK](https://learn.microsoft.com/zh-cn/java/openjdk/download) 安装包，安装时务必添加 JAVA_HOME 环境变量。安装结束后，打开终端输入：

```powershell
java -version
```

如果能正确的输出版本号即为安装成功。

### Linux

Ubuntu 平台安装 Java 直接使用包管理器即可，在终端输入：

```bash
sudo apt update && sudo apt upgrade
sudo apt install openjdk-21-jdk
```

安装完后，在终端输入：

```bash
java -version
```

如果能正确的输出版本号即为安装成功。

### 下载 Roboviz

下载 [Roboviz](https://github.com/magmaOffenburg/RoboViz/releases)，运行 `Roboviz.bat` 脚本 (Windows) 或 `Roboviz.sh`脚本 (Linux) 即可启动 Roboviz。

## Darkblue 球队代码安装构建

先安装基本的环境依赖：

```bash
sudo apt install cmake clang libgsl-dev python3-numpy python3-pybind11 python3-psutil python3-pip
```

更新 `pip` 的软件源地址：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

用 `pip` 安装 Python 模块：

```bash
pip3 install stable-baselines3 gym shimmy pyinstaller --break-system-packages
```

最后再拉取源码

```bash
git clone https://github.com/Robocon-NBUT/FCPCodebase.git
```

输入下面两行内容验证安装是否成功：

```bash
cd FCPCodebase
python3 Run_Utils.py
```

如果没有报错，就代表环境搭建完成。
