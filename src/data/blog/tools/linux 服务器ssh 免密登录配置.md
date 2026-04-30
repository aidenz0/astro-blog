---
title: linux 服务器ssh 免密登录配置
author: Aidenz
pubDatetime: 2025-06-06T04:06:31Z
slug: ssh-linux-server
featured: true
draft: false
tags:
  - linux
  - ssh
description: linux 服务器ssh 免密登录配置
---


> 最近租了一台香港的ubuntu服务器，下面是ssh 免密登录配置的记录 自己的电脑是macbook air m1

## 0. 查看本地是否有密钥

在终端中运行
```bash
ls -al ~/.ssh/
```

如果目录中有`id_rsa`和`id_rsa_pub`的文件，则本地有密钥，不需要进行第一步的生成密钥。

## 1. 在本地生成密钥

在终端中执行
```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```


- t rsa：指定RSA算法
    
- b 4096：设置密钥长度为4096位，更安全
    
- C：添加注释（通常为邮箱）
    

## 2. 把公钥复制到到服务器

- 方式一：使用`ssh-copy-id`
    
```bash
ssh-copy-id root@你的服务器IP
```

- 方式二：手动粘贴 执行下面的指令，复制输出的公钥
    ```bash
    cat ~/.ssh/id_rsa_pub.pub
    ```


登录服务器
```bash
ssh root@你的服务器IP
```



把复制公钥粘贴到`~/.ssh/authorized_keys`中 如果没有该目录和文件，要进行创建，并执行一下命令来设置权限：
```bash
chmod 700 ~/.ssh  
chmod 600 ~/.ssh/authorized_keys
```


## 3. 修改`/etc/ssh/sshd_config`

检查`PubkeyAuthentication`是yes 还是 no
```bash
grep -E "" /etc/ssh/sshd_config
```


将`PubkeyAuthentication`修改为yes
```bash
PubkeyAuthentication yes
```

## 4. 重启ssh服务
```bash
systemctl restart ssh
```
