# Anchor computation tool

这个repo原始版本有clio98提供，我在基础上修改了一下必要的配置步骤和教程。有任何问题欢迎提issue。


## 使用

将这个代码clone到和efficientdet同一个根目录的地方，这里指的是Yet-Another-Efficientdet的版本，然后传入你的project配置文件和数据根目录即可。

```
python3 anchor_computation_tool/anchor_inspector.py -project_name projects/taaa_4gpus_3x -dataset_path ./datasets/coco_taaa --annotations --anchors
```

请注意在你的配置里面添加字段：

```
no_gui: False
anchors: True
annotations: True
```

然后就可以开始inspect自己的anchor设置了。通过运行：

```
python3 anchor_computation_tool/kmeans_anchor_size.py ./datasets/coco_taaa
```

可以计算自己的初始anchor大小。
