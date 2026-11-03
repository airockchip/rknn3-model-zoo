# Mage-VL 视频测试数据

`pedestrian_area_cc0.mp4` 用于验证 Mage-VL C++ Demo 的 Frame 与 Codec 两种视频输入模式。

- 原始文件：`Video Codec Test pedestrian area 1080p25.y4m.webm`
- 来源：https://commons.wikimedia.org/wiki/File:Video_Codec_Test_pedestrian_area_1080p25.y4m.webm
- 作者：Taurus Media Technik
- 许可：CC0 1.0（可自由复制、修改和分发，无需署名）
- 原始文件 SHA-256：`bfadaa62cccb42db875d50bb842aa0964fbf72040432e4097c1df59e043e0c26`

为便于 RK3588/RK1828 测试，仓库内测试文件由原始 VP8 WebM 转码为 H.264 MP4：

- 分辨率：640×360
- 帧率：25 FPS
- 帧数：375
- 时长：15 秒
- 像素格式：YUV420P
- 测试文件 SHA-256：`46eecf7312108c8c81d678700c712c37fe5c2bbf127c96e676594b4b109176c3`

转码命令：

```bash
ffmpeg -i Video_Codec_Test_pedestrian_area_1080p25.y4m.webm -an \
  -vf scale=640:360:flags=lanczos -c:v libx264 -preset medium -crf 26 \
  -g 50 -keyint_min 50 -sc_threshold 0 -pix_fmt yuv420p -movflags +faststart \
  pedestrian_area_cc0.mp4
```
