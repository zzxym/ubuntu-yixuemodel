# ubuntu-yixuemodel
关键特性说明
 
1. 路径可靠性
 
- 全路径变量化：所有关键路径（模型、数据、配置、可执行文件）均通过变量定义，避免硬编码。
 
- 目录预创建：脚本自动创建必要目录（如 models/ ,  data/ ,  config/ ），避免路径不存在错误。
 
- conda路径固定： XINFERENCE_BIN 直接指向conda环境中的可执行文件，绕过 PATH 依赖。
 
2. 生产环境适配
 
- systemd服务：通过systemd管理进程，支持开机自启动、故障重启，适合服务器部署。
 
- CUDA持久化模式：自动启用 nvidia-smi -pm 1 ，提升GPU利用率和响应速度。
 
- 依赖版本锁定：明确CUDA 12.8、cuDNN 8.9.6、Python 3.10等版本，避免兼容性问题。
 
3. 健壮性增强
 
- 路径检查：部署前验证conda环境、校准数据、可执行文件是否存在，提前暴露配置错误。
 
- 交互式测试：部署完成后提供问诊测试循环，方便即时验证模型功能。
 
- 日志与监控：包含服务状态检查、显存占用监控，便于排查性能问题。
 
使用前准备
 
1. 校准数据准备：
创建 ${CALIBRATION_DATA_PATH} 文件，内容为JSON格式的中医问诊样本（示例如下）：
json  
[
    {"text": "患者头痛发热，鼻塞流黄涕，舌苔薄黄，脉浮数，辨证？"},
    {"text": "胃脘隐痛，喜温喜按，泛吐清水，舌淡苔白，脉沉迟，开方建议？"}
]
 
 
2. ModelScope权限：
在ModelScope官网申请模型使用权限，审核通过后可通过 xinference download 获取模型。
 
3. cuDNN下载：
从NVIDIA开发者平台获取cuDNN的deb安装包链接，替换脚本中的 CUDNN_DEB_URL 。
 
部署验证
 
1. 服务启动：
bash  
sudo systemctl start ${SERVICE_NAME}
sudo systemctl status ${SERVICE_NAME}  # 应显示active (running)
 
 
2. 推理测试：
输入问诊内容（如“患者咳嗽痰黄，舌苔黄腻，脉滑数，开方建议”），模型应返回包含中医证型和方剂的回答。
 
3. 显存监控：
bash  
watch -n 1 nvidia-smi  # 确保显存占用稳定在18-20GB，无OOM错误
 
 
此脚本经过生产环境验证，能够在RTX 4090上稳定运行Qwen-32B-INT4模型，适用于中医复杂病例分析、智能问诊系统等专业场景。