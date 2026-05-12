# Video Moment State Agent

一个用于“视频时刻定位”的最小可用原型，把自然语言 query 转成状态变化程序，再在视频状态轨迹中搜索并校验最匹配的时间段。

当前实现对应你图里的三步：

1. `State Program Induction`: `query -> P = (S, Delta, C)`
2. `State Grounding + Trace Search`: 在每一帧/时间片状态中找最符合程序的片段
3. `Program Consistency Verifier`: 检查完整性、顺序/因果、同一事件一致性

## 快速开始

```bash
python3 -m video_moment_agent induce "take milk from fridge"
python3 -m video_moment_agent locate examples/fridge_trace.json "take milk from fridge"
python3 -m unittest discover -s tests
```

示例输出会返回候选时间段、分数、命中的状态变化和校验结果。

## Trace JSON 格式

```json
{
  "video_id": "demo-fridge",
  "states": [
    {
      "t": 0.0,
      "state": {
        "fridge.state": "closed",
        "milk.loc": "inside"
      }
    }
  ]
}
```

## 下一步接真实视频

本仓库先实现核心状态程序与搜索器。真实视频接入建议：

1. 用 `ffmpeg` 生成 trace 脚手架：

```bash
python3 -m video_moment_agent scaffold-video ./video.mp4 ./runs/video_trace.json --fps 1
```

2. 用视觉模型把每帧的空 `state` 填成状态字典，例如 `fridge.state=open`、`milk.loc=outside`。
3. 调用 `locate` 搜索目标时刻。
