---
title: dify修改嵌入聊天框的power by图标
author: Aidenz
pubDatetime: 2025-09-11T04:06:31+08:00
slug:
featured: true
draft: false
tags:
  - dify
description: dify修改嵌入聊天框的power by图标
---

## 背景

最近近期公司需要使用dify框架开发一个公司网站的AI客服助手，但是嵌入网站时会显示powered by dify。有dify的logo。将dify 的logo修改为公司logo。
## 版本

dify版本：1.9.1

docker部署

## 解决方法

1. 停止dify服务`docker compose down`

2. 替换`/path/to/dify-main/web/public/logo/logo.svg`,替换为自己的logo文件

3. 在`/path/to/dify-main/docker`文件夹下，新建`run_dify.sh`脚本，

```bash

vim run_dify.sh

```

4. 将以下脚本复制进去

```shell

#!/bin/bash

# 退出时如果出错则中止脚本执行

set -e

  

echo "启动 docker compose..."

  

docker compose up -d --build

  

echo "切换目录到 /path/to/dify-main/web/public..."

cd /path/to/dify-main/web/public

  

echo "复制 logo 到容器 docker-web-1 中..."

docker cp logo docker-web-1:/app/web/public/

  

echo "重启容器 docker-web-1..."

docker restart docker-web-1

  

echo " 所有操作已完成！"

```

5. 使用脚本重启dify服务

```shell

./run_dify.sh

```

> 因为是将logo文件复制到容器中没有在镜像里，每次重启服务容器都会新建，所以需要用脚本，每次重启后都需要将logo文件再复制一遍。

> 每次启动使用脚本代替`docker compose up -d`