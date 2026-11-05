1105 改动：
现在，API 每次主要依据本次请求里的 messages 理解上下文；历史由本地 runner 维护并重新发送。

例如，一次“起飞后向前飞”的任务，消息列表会逐渐变成：
第一次请求
  system：系统提示词
  user：任务 + 初始观测 + 当前图片

第二次请求
  system：系统提示词
  user：任务 + 初始观测文字 + 初始观测图片
  assistant：{"tool":"takeoff"}
  user：任务 + 起飞结果 + 新观测 + 工具调用过程中的采样帧 + 当前图片

第三次请求
  system：系统提示词
  user：任务 + 初始观测文字
  assistant：{"tool":"takeoff"}
  user：任务 + 起飞结果 + 观测文字 + 起飞结束观测图片 (删除掉采样帧)
  assistant：{"tool":"fly_waypoint","waypoint":[...]}
  user：任务 + 飞行结果 + 新观测 + 工具调用过程中的采样帧 + 当前图片
