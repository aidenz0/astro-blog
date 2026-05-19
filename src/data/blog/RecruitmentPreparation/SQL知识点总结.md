---
title: SQL面试高频知识点全覆盖
author: Aidenz
pubDatetime: 2025-04-15T02:00:00Z
slug: sql-interview-guide
featured: false
draft: false
tags:
  - SQL
  - MySQL
  - PostgreSQL
  - 面试
description: 从数据库基础、表设计、索引原理到EXPLAIN执行计划分析，系统梳理SQL面试核心考点，附高频面试题和优化套路。
---

## 数据库基础

### MySQL vs PostgreSQL 核心区别

面试高频问题，需要能从多个维度对比：

| 维度      | MySQL                       | PostgreSQL                                 |
| ------- | --------------------------- | ------------------------------------------ |
| 事务      | InnoDB 支持完整 ACID，默认 RR 隔离级别 | 原生 MVCC，支持完整 ACID，默认 RC 隔离级别               |
| 索引      | InnoDB 聚簇索引 + B+Tree，支持全文索引 | B-Tree、Hash、GiST、GIN、BRIN 等多种索引类型          |
| 并发      | InnoDB 行锁 + MVCC + 间隙锁      | 原生 MVCC，无间隙锁，用 Serializable Snapshot 避免写偏斜 |
| JSON 支持 | JSON 类型，5.7+ 支持索引（虚拟列）      | JSON/JSONB 类型，JSONB 支持 GIN 索引，查询能力更强       |
| 扩展性     | 生态成熟，读写分离方案丰富               | 支持自定义类型、函数、操作符，扩展性更强                       |
| 适用场景    | 一般业务系统、高并发 OLTP             | 复杂查询、GIS、JSON 深度操作、分析型场景                   |

**面试话术**：一般业务选 MySQL，生态成熟运维成本低；涉及复杂查询、地理信息、JSON 深度操作或高并发分析场景选 PostgreSQL。

### 三大范式

- **1NF**：字段原子性，每个字段不可再拆分。比如「地址」应该拆成省、市、区、详细地址。
- **2NF**：在 1NF 基础上，非主键字段必须完全依赖主键，消除部分依赖。比如订单明细表中，商品名称应该依赖商品ID，而不是订单ID。
- **3NF**：在 2NF 基础上，非主键字段不能传递依赖于主键。比如学生表里不应该存院系名称，应该用院系ID关联。

**实际开发**：一般遵循 1-2NF，适当反范式化。比如订单表冗余用户姓名、商品名称，避免频繁 JOIN，用空间换时间。

### 字段选型原则

| 字段类型 | 选型建议                                                  |
| ---- | ----------------------------------------------------- |
| 主键   | 自增主键（简单场景）vs 雪花ID（分布式场景）。自增主键有序插入性能好，雪花ID全局唯一适合分库分表   |
| 整数   | 能用 INT 不用 BIGINT，节省存储和索引空间。用户量超 21 亿再用 BIGINT         |
| 字符串  | VARCHAR 按需设置长度，VARCHAR(255) 是常见上限。超长文本用 TEXT          |
| 时间   | DATETIME 存储范围更大（1000-9999），TIMESTAMP 自动时区转换且只到 2038 年 |

---

## 表设计实战

### 独立设计表的思路

设计一张表，按这个顺序思考：

1. **确定实体**：这个表代表什么？用户、订单、商品？
2. **确定字段**：这个实体有哪些属性？每个字段用什么类型？
3. **确定主键**：自增 vs 业务唯一键 vs 雪花ID
4. **确定关联**：和其他表的关系，用外键ID还是冗余字段？
5. **确定索引**：哪些字段会作为查询条件？需要联合索引吗？

### 练手：用户表 + 订单表

```sql
-- 用户表
CREATE TABLE `user` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '用户ID',
  `username` VARCHAR(64) NOT NULL COMMENT '用户名',
  `phone` VARCHAR(20) DEFAULT NULL COMMENT '手机号',
  `email` VARCHAR(128) DEFAULT NULL COMMENT '邮箱',
  `password_hash` VARCHAR(255) NOT NULL COMMENT '密码哈希',
  `status` TINYINT NOT NULL DEFAULT 1 COMMENT '状态: 1正常 0禁用',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_username` (`username`),
  UNIQUE KEY `uk_phone` (`phone`),
  KEY `idx_email` (`email`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户表';

-- 订单表
CREATE TABLE `order` (
  `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '订单ID',
  `order_no` VARCHAR(32) NOT NULL COMMENT '订单号',
  `user_id` BIGINT UNSIGNED NOT NULL COMMENT '用户ID',
  `user_name` VARCHAR(64) NOT NULL COMMENT '用户名(冗余)',
  `total_amount` DECIMAL(10,2) NOT NULL COMMENT '订单总金额',
  `status` TINYINT NOT NULL DEFAULT 0 COMMENT '状态: 0待付款 1已付款 2已发货 3已完成 4已取消',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_order_no` (`order_no`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_status_created` (`status`, `created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='订单表';
```

### 外键要不要用

**工作中尽量不用数据库外键**，理由：

- 外键会增加写入开销（每次 INSERT/UPDATE 都要检查引用完整性）
- 分库分表场景下外键无法跨库
- 删除/更新父记录时级联操作容易误删数据
- 程序层面控制关联更灵活

### 反范式化（字段冗余）适用场景

- 订单表冗余 `user_name`，避免查订单时 JOIN 用户表
- 商品快照表，下单时把商品信息完整冗余到订单明细
- 统计字段冗余，比如文章表冗余评论数、点赞数，避免 COUNT 查询

---

## 索引

### 索引类型

| 索引类型 | 说明                              |
| ---- | ------------------------------- |
| 普通索引 | 最基础的索引，加速查询，允许重复值               |
| 唯一索引 | 保证列值唯一，允许 NULL                  |
| 主键索引 | 特殊的唯一索引，不允许 NULL，InnoDB 下就是聚簇索引 |
| 联合索引 | 多列组合索引，遵循最左匹配原则                 |
| 覆盖索引 | 查询字段全部在索引中，不需要回表                |
| 全文索引 | 用于文本搜索，MySQL 5.6+ InnoDB 支持     |

### InnoDB 聚簇索引 vs 二级索引

**聚簇索引（主键索引）**：

- 叶子节点存储的是**完整的行数据**
- 一张表只能有一个聚簇索引（就是主键索引）
- 数据按主键顺序物理存储
- 主键查询性能最好，因为直接拿到数据

**二级索引（非聚簇索引）**：

- 叶子节点存储的是**主键值**
- 通过二级索引查到主键后，还需要**回表**去聚簇索引查完整数据
- 如果查询字段全部在二级索引中（覆盖索引），则不需要回表

**面试高频**：为什么 InnoDB 表建议用自增主键？

- 自增主键保证插入有序，不会频繁页分裂
- 主键长度小（BIGINT 8字节），二级索引叶子节点存储主键值，主键越小索引越紧凑
- 如果没有主键，InnoDB 会选一个唯一非空索引，都没有则生成隐藏的 6 字节 ROW_ID

### 联合索引最左匹配原则

联合索引 `(a, b, c)` 的 B+Tree 按 a → b → c 的顺序排序。

```sql
-- 走索引的情况
WHERE a = 1                          -- ✅ 使用 a
WHERE a = 1 AND b = 2               -- ✅ 使用 a, b
WHERE a = 1 AND b = 2 AND c = 3    -- ✅ 使用 a, b, c
WHERE a = 1 AND b > 2               -- ✅ a 等值 + b 范围
WHERE a = 1 ORDER BY b              -- ✅ a 等值 + b 排序

-- 不走索引（或只走部分）的情况
WHERE b = 2                          -- ❌ 跳过了 a
WHERE b = 2 AND c = 3               -- ❌ 跳过了 a
WHERE a = 1 AND c = 3               -- ⚠️ 只用到 a，c 无法走索引（中间断了）
WHERE a = 1 AND b LIKE '%xx'        -- ⚠️ a 走索引，b 范围查询后 c 无法走索引
```

**原理**：B+Tree 先按 a 排序，a 相同再按 b 排序，b 相同再按 c 排序。跳过 a 直接查 b，树的排序规则对 b 无意义，无法利用索引。

### 索引失效场景（背熟）

| 失效场景 | 示例 | 原因 |
|---------|------|------|
| LIKE 左模糊 | `WHERE name LIKE '%张'` | B+Tree 无法从中间开始匹配 |
| 隐式类型转换 | `WHERE phone = 13800000000`（phone 是 VARCHAR） | MySQL 会把 phone 转成数字，相当于对列做函数 |
| OR 连接无索引列 | `WHERE a = 1 OR b = 2`（b 无索引） | 只要有一个条件无索引就全表扫描 |
| NOT IN / IS NOT NULL | `WHERE id NOT IN (1,2,3)` | 优化器可能认为全表扫描更快 |
| 索引列做运算/函数 | `WHERE YEAR(created_at) = 2026` | 对列做函数，B+Tree 存的是原始值 |
| 联合索引不满足最左 | `WHERE b = 2`（索引是 (a,b)） | 跳过了最左列 |

**优化建议**：

- `LIKE '%张'` → 改用全文索引或搜索引擎
- 隐式转换 → 保证类型一致，字符串加引号
- `WHERE YEAR(created_at) = 2026` → 改为 `WHERE created_at >= '2026-01-01' AND created_at < '2027-01-01'`

---

## SQL 基础语句

### 增删改查

```sql
-- 插入
INSERT INTO `user` (username, phone, password_hash) VALUES ('张三', '13800000000', 'hash...');

-- 更新
UPDATE `user` SET status = 0 WHERE id = 1;

-- 删除
DELETE FROM `user` WHERE id = 1;

-- 查询
SELECT id, username, phone FROM `user` WHERE status = 1 ORDER BY created_at DESC LIMIT 10;
```

### 多表联查

```sql
-- LEFT JOIN：查所有用户及其订单（包括没有订单的用户）
SELECT u.id, u.username, o.order_no, o.total_amount
FROM `user` u
LEFT JOIN `order` o ON u.id = o.user_id
WHERE u.status = 1;

-- INNER JOIN：只查有订单的用户
SELECT u.id, u.username, COUNT(o.id) AS order_count, SUM(o.total_amount) AS total
FROM `user` u
INNER JOIN `order` o ON u.id = o.user_id
GROUP BY u.id, u.username
HAVING order_count > 5
ORDER BY total DESC;
```

### 子查询

```sql
-- 查订单金额大于平均值的订单
SELECT * FROM `order`
WHERE total_amount > (SELECT AVG(total_amount) FROM `order`);

-- 查下过单的用户（用 EXISTS 比 IN 更高效）
SELECT * FROM `user` u
WHERE EXISTS (SELECT 1 FROM `order` o WHERE o.user_id = u.id);
```

### 什么是慢查询

执行时间超过 `long_query_time`（默认 10 秒）的 SQL。慢查询日志记录在 `slow_query_log_file` 中。

慢的原因通常是：
- 没走索引，全表扫描
- 索引设计不合理
- 数据量太大
- 锁等待
- 返回数据量过多

---

## EXPLAIN 执行计划

### 关键字段详解

```sql
EXPLAIN SELECT * FROM `order` WHERE user_id = 1 AND status = 1;
```

| 字段                | 含义                                                              | 重点关注                                |
| ----------------- | --------------------------------------------------------------- | ----------------------------------- |
| **id**            | 查询序号，id 相同从上往下执行，id 不同值越大越先执行                                   | 子查询执行顺序                             |
| **select_type**   | 查询类型：SIMPLE（简单查询）、PRIMARY（外层）、SUBQUERY（子查询）、DERIVED（派生表）        | 区分简单查询和复杂查询                         |
| **type**          | 访问类型，从好到差：`system > const > eq_ref > ref > range > index > ALL` | **最重要**，决定查询效率                      |
| **possible_keys** | 可能使用的索引                                                         | 对比 key 看是否选对了索引                     |
| **key**           | 实际使用的索引                                                         | NULL 表示没走索引                         |
| **key_len**       | 索引使用的字节长度                                                       | 联合索引中判断用到了几个字段                      |
| **rows**          | 预估扫描行数                                                          | 越小越好                                |
| **Extra**         | 额外信息                                                            | 看是否有 Using filesort、Using temporary |

### type 字段详解（重点）

```
ALL < index < range < ref < eq_ref < const < system
```

- **ALL**：全表扫描，最差。需要加索引或优化 SQL。
- **index**：全索引扫描，比 ALL 好一点。比如 `SELECT id FROM table`，id 是主键。
- **range**：索引范围扫描，常见于 `BETWEEN`、`>`、`<`、`IN`。
- **ref**：非唯一索引等值查询，可能返回多行。比如 `WHERE user_id = 1`（user_id 是普通索引）。
- **eq_ref**：唯一索引等值查询，最多返回一行。常见于 JOIN 中主键关联。
- **const**：主键或唯一索引等值查询，最多一行。比如 `WHERE id = 1`。
- **system**：表只有一行，特殊场景。

### Extra 字段详解

| Extra 值               | 含义                       | 优化方向                   |
| --------------------- | ------------------------ | ---------------------- |
| Using index           | 覆盖索引，不需要回表               | 理想状态                   |
| Using where           | 在存储引擎层过滤后，Server 层还需要再过滤 | 检查是否可以加索引              |
| Using filesort        | 额外排序操作，没有利用索引排序          | 优化 ORDER BY 字段，加联合索引   |
| Using temporary       | 使用了临时表                   | 优化 GROUP BY 或 DISTINCT |
| Using index condition | 索引下推（ICP），在存储引擎层提前过滤     | MySQL 5.6+ 优化，是好事      |

### EXPLAIN 实操分析

```sql
-- 场景1：没走索引
EXPLAIN SELECT * FROM `order` WHERE YEAR(created_at) = 2026;
-- type: ALL, key: NULL → 全表扫描，索引失效

-- 场景2：走索引
EXPLAIN SELECT * FROM `order` WHERE created_at >= '2026-01-01' AND created_at < '2027-01-01';
-- type: range, key: idx_created_at → 走了范围索引

-- 场景3：联合索引使用情况
EXPLAIN SELECT * FROM `order` WHERE user_id = 1 AND status = 1;
-- key_len 可以判断用了联合索引的几个字段
```

---

## SQL 优化套路（面试直接说）

### 通用优化手段

1. **加合适的索引**
   - 高频查询字段加索引
   - 联合索引遵循最左匹配，把区分度高的字段放前面
   - 避免冗余索引：`(a,b)` 已经包含了 `(a)` 的功能

2. **避免索引失效写法**
   - 不用 `LIKE '%xx'`、不对索引列做函数/运算
   - 字符串条件加引号避免隐式转换
   - 用 `EXISTS` 替代 `IN`（大子查询场景）

3. **禁止 `SELECT *`**
   - 只查需要的字段，配合覆盖索引避免回表
   - 减少网络传输和内存占用

4. **大表分页优化**

```sql
-- 慢：LIMIT 100000, 10 会扫描 100010 行再丢弃前 100000 行
SELECT * FROM `order` ORDER BY id LIMIT 100000, 10;

-- 优化1：用子查询先定位 ID
SELECT * FROM `order` WHERE id >= (SELECT id FROM `order` ORDER BY id LIMIT 100000, 1) LIMIT 10;

-- 优化2：记住上次的最大 ID（游标分页）
SELECT * FROM `order` WHERE id > 100000 ORDER BY id LIMIT 10;

-- 优化3：延迟关联
SELECT o.* FROM `order` o
INNER JOIN (SELECT id FROM `order` ORDER BY id LIMIT 100000, 10) t ON o.id = t.id;
```

5. **小表驱动大表**
   - JOIN 时小表做驱动表（放在 LEFT JOIN 左边或 INNER JOIN 优化器自动选择）
   - `IN` 适合子查询结果集小的场景，`EXISTS` 适合外表小的场景

6. **大表分库分表 + 冷热数据拆分**
   - 按用户ID分库，按时间分表
   - 历史订单归到冷数据表，主表只保留近 3 个月

7. **合理设置字段类型**
   - 能用 TINYINT 不用 INT，能用 DATE 不用 DATETIME
   - VARCHAR 按实际需要设置长度，不要无脑 VARCHAR(255)

---

## MySQL vs PostgreSQL 使用场景

| 场景 | 推荐 | 理由 |
|------|------|------|
| 一般业务系统（CRUD） | MySQL | 生态成熟，运维工具多，读写分离方案完善 |
| 复杂查询、窗口函数 | PostgreSQL | 原生支持更丰富的 SQL 语法 |
| 地理信息（GIS） | PostgreSQL + PostGIS | MySQL 的 GIS 能力远不如 PG |
| JSON 深度查询 | PostgreSQL | JSONB 类型 + GIN 索引，查询性能和功能都更强 |
| 高并发分析（OLAP） | PostgreSQL | 支持并行查询、物化视图 |
| 分布式 / 海量数据 | TiDB / CockroachDB | 兼容 MySQL/PG 协议的分布式数据库 |

---

## 面试高频问题速查

**Q：聚簇索引和二级索引的区别？**
聚簇索引叶子存行数据，二级索引叶子存主键值。通过二级索引查到主键后需要回表。覆盖索引可以避免回表。

**Q：联合索引 (a,b,c)，`WHERE b=1 AND c=2` 走不走索引？**
不走。违反最左匹配原则，跳过了最左列 a。

**Q：索引失效的常见场景？**
LIKE 左模糊、隐式类型转换、索引列做函数/运算、OR 连接无索引列、联合索引不满足最左。

**Q：EXPLAIN 中 type 从好到差？**
`system > const > eq_ref > ref > range > index > ALL`，ALL 是全表扫描。

**Q：EXPLAIN 中 Extra 出现 Using filesort 怎么优化？**
给 ORDER BY 的字段加索引，最好是联合索引中包含排序字段。

**Q：LIMIT 100000, 10 为什么慢？怎么优化？**
因为要扫描 100010 行再丢弃前 100000 行。优化方案：游标分页（记住上次最大ID）、子查询定位、延迟关联。

**Q：为什么工作中不用数据库外键？**
外键增加写入开销、无法跨库、级联操作有风险，程序控制更灵活。

**Q：什么时候用反范式化？**
频繁 JOIN 的字段冗余、下单时的商品快照、统计字段（评论数、点赞数）。
