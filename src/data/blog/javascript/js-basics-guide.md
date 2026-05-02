---
title: JavaScript 基础完全指南 - 从零开始让网页动起来
author: Aidenz
pubDatetime: 2026-05-02T07:00:00Z
slug: js-basics-guide
featured: false
draft: false
tags:
  - JavaScript
  - 前端
  - 入门教程
description: 全面系统地介绍 JavaScript 基础知识，包括语法、函数、DOM 操作、事件处理等核心概念，适合初学者查缺补漏。
---

## 引言

JavaScript 是一种动态的、弱类型的编程语言，是网页开发中实现交互功能的核心技术。本文将全面介绍 JavaScript 的核心知识点。

## 一、JavaScript 导入方式

### 1.1 三种导入方式

```html
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <title>JavaScript 导入方式</title>

  <!-- 方式1：head 中的内联脚本 -->
  <script>
    console.log("这是 head 中的内联脚本");
  </script>

  <!-- 方式2：引入外部 JS 文件（推荐） -->
  <script src="js/helloworld.js"></script>
</head>
<body>
  <h1>JavaScript 导入方式</h1>

  <!-- 方式3：body 中的内联脚本（推荐放在 body 底部） -->
  <script>
    console.log("这是 body 中的内联脚本");
  </script>
</body>
</html>
```

**导入方式对比：**

| 方式 | 优点 | 缺点 | 推荐场景 |
|------|------|------|----------|
| head 中内联 | 立即执行 | 阻塞页面渲染 | 不推荐 |
| head 中外部 | 分离关注点 | 阻塞页面渲染（除非用 defer/async） | 工具函数库 |
| body 底部 | 不阻塞渲染 | 可能延迟交互执行 | 页面主脚本 |

### 1.2 script 标签属性

```html
<!-- defer：延迟执行，按顺序执行（DOMContentLoaded 之前） -->
<script defer src="script1.js"></script>
<script defer src="script2.js"></script>
<!-- 保证 script1 在 script2 之前执行 -->

<!-- async：异步加载，加载完立即执行（不保证顺序） -->
<script async src="analytics.js"></script>

<!-- type：指定脚本类型 -->
<script type="module" src="app.js"></script> <!-- ES6 模块 -->
<script type="text/babel" src="app.jsx"></script> <!-- JSX -->

<!-- nomodule：不支持模块的浏览器回退 -->
<script nomodule src="legacy.js"></script>

<!-- crossorigin：跨域脚本 -->
<script src="https://example.com/script.js" crossorigin="anonymous"></script>

<!-- integrity：完整性校验（防止篡改） -->
<script src="script.js" integrity="sha384-..."></script>
```

**defer vs async：**

```
页面加载过程对比：

普通（无属性）：
HTML解析 → 遇到script → 暂停解析 → 下载script → 执行script → 继续解析HTML

defer：
HTML解析（并行下载script） → HTML解析完成 → 按顺序执行script → DOMContentLoaded

async：
HTML解析（并行下载script） → 下载完成立即执行 → 继续解析HTML
```

## 二、JavaScript 基本语法

### 2.1 变量声明

```javascript
// var：函数作用域，可重复声明，可提升（不推荐）
var x;
console.log(x); // undefined

// let：块级作用域，不可重复声明，不可提升（推荐）
let y;
console.log(y); // undefined

// const：块级作用域，必须初始化，不可重新赋值（推荐）
const PI = 3.14;
// PI = 3.15; // 报错：Assignment to constant variable

// 变量赋值
let name = "John";
let emptyValue = null;
let notDefined;
console.log(notDefined); // undefined
```

**var、let、const 对比：**

| 特性 | var | let | const |
|------|-----|-----|-------|
| 作用域 | 函数作用域 | 块级作用域 | 块级作用域 |
| 重复声明 | 允许 | 不允许 | 不允许 |
| 重新赋值 | 允许 | 允许 | 不允许 |
| 变量提升 | 是 | 否 | 否 |
| 暂时性死区 | 无 | 有 | 有 |

### 2.2 数据类型

```javascript
// 基本数据类型（Primitive）
let str = "Hello";           // String
let num = 42;                // Number
let float = 3.14;            // Number
let bool = true;             // Boolean
let empty = null;            // Null
let nothing = undefined;     // Undefined
let symbol = Symbol("id");   // Symbol (ES6)
let bigint = 123n;           // BigInt (ES2020)

// 引用数据类型
let obj = { name: "John" };  // Object
let arr = [1, 2, 3];         // Array（本质是 Object）
let func = function() {};    // Function（本质是 Object）

// 类型检测
console.log(typeof str);      // "string"
console.log(typeof num);      // "number"
console.log(typeof bool);     // "boolean"
console.log(typeof obj);      // "object"
console.log(typeof arr);      // "object"（注意！）
console.log(typeof null);     // "object"（历史遗留问题）
console.log(typeof func);     // "function"

// 准确检测数组
console.log(Array.isArray(arr)); // true
console.log(arr instanceof Array); // true
```

### 2.3 类型转换

```javascript
// 字符串转换
let num = 123;
let str = String(num);      // "123"
let str2 = num.toString();  // "123"
let str3 = "" + num;        // "123"（隐式转换）

// 数字转换
let str = "123";
let num = Number(str);      // 123
let num2 = parseInt(str);   // 123
let num3 = parseFloat("3.14"); // 3.14

// 布尔转换
Boolean(0);         // false
Boolean("");        // false
Boolean(null);      // false
Boolean(undefined); // false
Boolean(NaN);       // false
Boolean("false");   // true（非空字符串都是 true）

// 隐式转换的坑
console.log(1 + "2");    // "12"（拼接）
console.log(1 - "2");    // -1（数字运算）
console.log(1 * "2");    // 2
console.log(1 == "1");   // true（类型转换）
console.log(1 === "1");  // false（严格相等）
```

### 2.4 条件语句

```javascript
// if...else
let age = 30;
if (age > 18) {
  console.log("成年人");
} else if (age > 12) {
  console.log("青少年");
} else {
  console.log("儿童");
}

// 三元运算符
let message = age >= 18 ? "可以投票" : "不能投票";

// switch...case
let day = "Monday";
switch (day) {
  case "Monday":
    console.log("周一");
    break;
  case "Tuesday":
    console.log("周二");
    break;
  default:
    console.log("其他");
}

// 逻辑运算符短路
let name = null;
let displayName = name || "匿名"; // "匿名"

let config = {};
let value = config?.setting?.value; // 可选链，避免报错
```

### 2.5 循环语句

```javascript
// while 循环
let i = 0;
while (i < 5) {
  console.log("while:", i);
  i++;
}

// do...while 循环（至少执行一次）
let j = 0;
do {
  console.log("do-while:", j);
  j++;
} while (j < 5);

// for 循环
for (let k = 0; k < 5; k++) {
  console.log("for:", k);
}

// for...in（遍历对象属性）
const obj = { a: 1, b: 2, c: 3 };
for (let key in obj) {
  console.log(key, obj[key]);
}

// for...of（遍历可迭代对象）
const arr = ["Apple", "Banana", "Cherry"];
for (let fruit of arr) {
  console.log(fruit);
}

// 数组方法遍历
arr.forEach((fruit, index) => {
  console.log(index, fruit);
});

// break 和 continue
for (let n = 0; n < 10; n++) {
  if (n === 5) {
    break; // 跳出循环
  }
  console.log(n);
}

for (let n = 0; n < 10; n++) {
  if (n === 5) {
    continue; // 跳过本次循环
  }
  console.log(n);
}
```

## 三、JavaScript 函数

### 3.1 函数定义方式

```javascript
// 1. 函数声明（会被提升）
function sayHello() {
  console.log("Hello World!");
  return "Hello World!";
}

// 2. 函数表达式（不会被提升）
const sayHi = function(greeting, name) {
  console.log(greeting + " " + name);
};

// 3. 箭头函数（ES6，无自己的 this）
const sayBye = () => {
  console.log("Bye World!");
};

// 简化的箭头函数
const add = (a, b) => a + b; // 单行可省略 return 和 {}
const square = x => x * x;   // 单参数可省略 ()

// 4. 构造函数（不推荐）
const sayHey = new Function("console.log('Hey');");
```

### 3.2 函数参数

```javascript
// 默认参数
function greet(name = "朋友") {
  console.log(`你好，${name}！`);
}
greet(); // 你好，朋友！

// 剩余参数
function sum(...numbers) {
  return numbers.reduce((total, num) => total + num, 0);
}
console.log(sum(1, 2, 3, 4)); // 10

// 解构参数
function createUser({ name, age, city = "北京" }) {
  return { name, age, city };
}
console.log(createUser({ name: "张三", age: 25 })); // {name: "张三", age: 25, city: "北京"}

// 参数校验
function divide(a, b) {
  if (typeof a !== "number" || typeof b !== "number") {
    throw new TypeError("参数必须是数字");
  }
  if (b === 0) {
    throw new Error("除数不能为零");
  }
  return a / b;
}
```

### 3.3 作用域与闭包

```javascript
// 全局作用域
let globalVar = "全局";

function outer() {
  // 函数作用域
  let outerVar = "外部";

  function inner() {
    // 内层函数可访问外层变量
    let innerVar = "内部";
    console.log(globalVar); // "全局"
    console.log(outerVar);  // "外部"
    console.log(innerVar);  // "内部"
  }

  inner();
  // console.log(innerVar); // 报错：内层变量外层不可访问
}

// 闭包：函数访问其词法作用域外的变量
function createCounter() {
  let count = 0;
  return function() {
    count++;
    return count;
  };
}

const counter = createCounter();
console.log(counter()); // 1
console.log(counter()); // 2
console.log(counter()); // 3

// 闭包的实际应用：数据私有化
function createPerson(name) {
  let _name = name; // 私有变量

  return {
    getName() {
      return _name;
    },
    setName(newName) {
      _name = newName;
    }
  };
}

const person = createPerson("张三");
console.log(person.getName()); // "张三"
person.setName("李四");
console.log(person.getName()); // "李四"
// console.log(person._name); // undefined，无法直接访问
```

### 3.4 this 关键字

```javascript
// 全局 this
console.log(this === window); // true（浏览器中）

// 对象方法中的 this
const person = {
  name: "张三",
  sayName() {
    console.log(this.name); // "张三"
  }
};

// 箭头函数没有自己的 this
const person2 = {
  name: "李四",
  sayName: () => {
    console.log(this.name); // undefined（指向外层 this）
  }
};

// call/apply/bind 改变 this
function greet() {
  console.log(`你好，${this.name}`);
}

const obj1 = { name: "王五" };
const obj2 = { name: "赵六" };

greet.call(obj1);  // 你好，王五
greet.apply(obj2); // 你好，赵六

const boundGreet = greet.bind({ name: "孙七" });
boundGreet(); // 你好，孙七
```

## 四、JavaScript 对象与数组

### 4.1 对象基础

```javascript
// 对象字面量
const person = {
  name: "张三",
  age: 25,
  city: "北京",
  greet() {
    console.log(`我是${this.name}`);
  }
};

// 访问属性
console.log(person.name);      // 点表示法
console.log(person["age"]);    // 括号表示法（可计算）

// 添加/修改属性
person.email = "zhangsan@example.com";
person.age = 26;

// 删除属性
delete person.email;

// 对象解构
const { name, age } = person;
console.log(name, age); // "张三" 26

// 对象展开
const person2 = { ...person, city: "上海" };
console.log(person2); // {name: "张三", age: 26, city: "上海"}

// Object 常用方法
Object.keys(person);       // ["name", "age", "city"]
Object.values(person);     // ["张三", 25, "北京"]
Object.entries(person);    // [["name", "张三"], ["age", 25], ...]
Object.assign({}, person, { age: 30 }); // 合并对象
```

### 4.2 数组基础

```javascript
// 创建数组
const arr1 = [1, 2, 3];
const arr2 = new Array(1, 2, 3);
const arr3 = Array.of(1, 2, 3);

// 访问元素
console.log(arr1[0]); // 1
console.log(arr1[arr1.length - 1]); // 最后一个元素

// 数组解构
const [first, second, ...rest] = arr1;
console.log(first, second, rest); // 1 2 [3]

// 数组展开
const arr4 = [...arr1, 4, 5];
console.log(arr4); // [1, 2, 3, 4, 5]
```

### 4.3 数组方法

```javascript
const arr = [1, 2, 3, 4, 5];

// 添加/删除元素
arr.push(6);           // 尾部添加，返回新长度
arr.pop();             // 尾部删除，返回删除元素
arr.unshift(0);        // 头部添加
arr.shift();           // 头部删除
arr.splice(2, 1, 99);  // 从索引2删除1个，插入99

// 查找
arr.indexOf(3);        // 返回索引，找不到返回 -1
arr.includes(3);       // 返回 boolean
arr.find(x => x > 3);  // 返回第一个满足条件的元素
arr.findIndex(x => x > 3); // 返回索引

// 遍历
arr.forEach((item, index) => {
  console.log(index, item);
});

// 转换
arr.map(x => x * 2);           // [2, 4, 6, 8, 10]
arr.filter(x => x > 2);        // [3, 4, 5]
arr.reduce((sum, x) => sum + x, 0); // 15（求和）

// 检测
arr.every(x => x > 0);  // true（全部满足）
arr.some(x => x > 3);   // true（有满足的）

// 排序
arr.sort((a, b) => a - b); // 升序
arr.reverse();             // 反转

// 其他
arr.slice(1, 3);      // 截取 [2, 3]（不修改原数组）
arr.join("-");        // "1-2-3-4-5"
arr.concat([6, 7]);   // [1, 2, 3, 4, 5, 6, 7]
arr.toString();       // "1,2,3,4,5"
```

## 五、DOM 操作

DOM（Document Object Model）是 JavaScript 操作 HTML 文档的接口。

### 5.1 获取元素

```javascript
// 通过 ID 获取（单个）
const element1 = document.getElementById("box1");

// 通过类名获取（多个，返回 HTMLCollection）
const elements2 = document.getElementsByClassName("box2");

// 通过标签名获取（多个）
const elements3 = document.getElementsByTagName("div");

// 通过选择器获取（单个）
const element4 = document.querySelector("#box1");
const element5 = document.querySelector(".box2");

// 通过选择器获取（多个，返回 NodeList）
const elements6 = document.querySelectorAll(".box2");

// HTMLCollection vs NodeList
// HTMLCollection：动态（随 DOM 变化），类数组
// NodeList：静态（快照），forEach 方法
```

### 5.2 修改内容

```javascript
const element = document.getElementById("box1");

// 修改文本内容
element.textContent = "新文本";
element.innerText = "新文本"; // 类似，但考虑样式

// 修改 HTML 内容
element.innerHTML = "<strong>加粗文本</strong>";

// 获取表单值
const input = document.querySelector("input");
console.log(input.value);

// 修改表单值
input.value = "新值";
```

### 5.3 修改样式

```javascript
const element = document.getElementById("box3");

// 直接修改 style
element.style.backgroundColor = "red";
element.style.fontSize = "30px";

// CSS 属性名转换：background-color → backgroundColor
element.style.cssText = "background-color: red; font-size: 30px;";

// 操作 class（推荐）
element.className = "box active";
element.classList.add("active");
element.classList.remove("box");
element.classList.toggle("active"); // 有则删，无则加
element.classList.contains("active"); // 检查是否存在

// 操作属性
element.setAttribute("data-id", "123");
element.getAttribute("data-id");
element.removeAttribute("data-id");
element.hasAttribute("data-id"); // boolean

// 自定义数据属性
element.dataset.id = "123";
element.dataset.userType = "admin";
console.log(element.dataset.userId);
```

### 5.4 创建和删除元素

```javascript
// 创建元素
const div = document.createElement("div");
div.textContent = "新元素";
div.className = "new-element";

// 添加到文档
document.body.appendChild(div);

// 插入到指定位置
const parent = document.getElementById("parent");
const reference = document.getElementById("reference");
parent.insertBefore(div, reference);

// 替换元素
parent.replaceChild(newElement, oldElement);

// 删除元素
parent.removeChild(div);
// 或
div.remove(); // 直接删除自己

// 克隆元素
const cloned = div.cloneNode(true); // true 表示深度克隆（包含子元素）
```

### 5.5 DOM 遍历

```javascript
const element = document.querySelector(".box");

// 父节点
element.parentNode;       // 直接父节点
element.parentElement;    // 直接父元素（不含文档节点）

// 子节点
element.childNodes;       // 所有子节点（包含文本节点）
element.children;         // 所有子元素（不含文本节点）
element.firstChild;       // 第一个子节点
element.firstElementChild; // 第一个子元素
element.lastChild;
element.lastElementChild;

// 兄弟节点
element.nextSibling;      // 下一个兄弟节点
element.nextElementSibling; // 下一个兄弟元素
element.previousSibling;
element.previousElementSibling;
```

## 六、JavaScript 事件

事件是文档或浏览器中发生的特定交互瞬间。

### 6.1 事件绑定方式

```html
<!-- 方式1：HTML 属性（不推荐） -->
<button onclick="handleClick()">点击我</button>

<!-- 方式2：DOM 属性 -->
<button id="btn">点击我</button>
<script>
  const btn = document.getElementById("btn");
  btn.onclick = function() {
    console.log("按钮被点击");
  };
</script>

<!-- 方式3：addEventListener（推荐） -->
<button id="btn2">点击我</button>
<script>
  const btn2 = document.getElementById("btn2");
  btn2.addEventListener("click", function() {
    console.log("按钮被点击");
  });
</script>
```

**三种方式对比：**

| 方式 | 优点 | 缺点 |
|------|------|------|
| HTML 属性 | 简单直接 | HTML 与 JS 耦合 |
| DOM 属性 | 简单 | 只能绑定一个处理函数 |
| addEventListener | 可绑定多个、可控制阶段 | 写法稍复杂 |

### 6.2 常用事件类型

```javascript
// 鼠标事件
element.addEventListener("click", handler);       // 点击
element.addEventListener("dblclick", handler);    // 双击
element.addEventListener("mousedown", handler);   // 按下
element.addEventListener("mouseup", handler);     // 松开
element.addEventListener("mouseover", handler);   // 鼠标移入（冒泡）
element.addEventListener("mouseout", handler);    // 鼠标移出（冒泡）
element.addEventListener("mouseenter", handler);  // 鼠标移入（不冒泡）
element.addEventListener("mouseleave", handler);  // 鼠标移出（不冒泡）
element.addEventListener("mousemove", handler);   // 移动

// 键盘事件
element.addEventListener("keydown", handler);     // 按下
element.addEventListener("keyup", handler);       // 松开
element.addEventListener("keypress", handler);    // 按住（已废弃）

// 表单事件
element.addEventListener("focus", handler);       // 获得焦点
element.addEventListener("blur", handler);        // 失去焦点
element.addEventListener("change", handler);      // 值改变
element.addEventListener("input", handler);       // 输入时
element.addEventListener("submit", handler);      // 提交表单

// 文档/窗口事件
window.addEventListener("load", handler);         // 加载完成
window.addEventListener("resize", handler);       // 大小改变
window.addEventListener("scroll", handler);       // 滚动
document.addEventListener("DOMContentLoaded", handler); // DOM 加载完成
```

### 6.3 事件对象

```javascript
button.addEventListener("click", function(event) {
  // 阻止默认行为
  event.preventDefault();

  // 阻止事件传播
  event.stopPropagation();

  // 事件目标
  console.log(event.target);       // 实际触发元素
  console.log(event.currentTarget); // 绑定事件的元素

  // 鼠标位置
  console.log(event.clientX, event.clientY); // 相对视口
  console.log(event.pageX, event.pageY);     // 相对文档

  // 按键信息
  console.log(event.key);      // 按键字符
  console.log(event.code);     // 按键代码
  console.log(event.ctrlKey);  // 是否按下 Ctrl
  console.log(event.shiftKey); // 是否按下 Shift
});
```

### 6.4 事件冒泡与捕获

```javascript
// 事件传播阶段：捕获 → 目标 → 冒泡
// 默认在冒泡阶段触发

div.addEventListener("click", handler, {
  capture: true,  // 在捕获阶段触发
  once: true,     // 只触发一次
  passive: true   // 不会调用 preventDefault（提升滚动性能）
});

// 事件委托（利用冒泡）
ul.addEventListener("click", function(event) {
  if (event.target.tagName === "LI") {
    console.log("点击了 li:", event.target.textContent);
  }
});
```

## 七、ES6+ 新特性

### 7.1 let 和 const

```javascript
// 块级作用域
{
  let x = 10;
  const y = 20;
  console.log(x, y); // 10 20
}
// console.log(x); // ReferenceError

// const 声明的对象属性可修改
const obj = { name: "张三" };
obj.name = "李四"; // 可以
// obj = {}; // 报错
```

### 7.2 模板字符串

```javascript
const name = "张三";
const age = 25;

// 多行字符串
const html = `
  <div>
    <h1>${name}</h1>
    <p>年龄：${age}</p>
  </div>
`;

// 表达式
const result = `明年 ${age + 1} 岁`;
```

### 7.3 解构赋值

```javascript
// 数组解构
const [a, b, ...rest] = [1, 2, 3, 4, 5];
console.log(a, b, rest); // 1 2 [3, 4, 5]

// 对象解构
const { name, age, city = "北京" } = { name: "张三", age: 25 };

// 函数参数解构
function greet({ name, age }) {
  console.log(`${name} 今年 ${age} 岁`);
}

// 交换变量
let x = 1, y = 2;
[x, y] = [y, x];
```

### 7.4 箭头函数

```javascript
// 基本语法
const add = (a, b) => a + b;

// 等价于
function add(a, b) {
  return a + b;
}

// 注意：箭头函数没有自己的 this
const obj = {
  name: "张三",
  // 错误示例
  greetWrong: () => {
    console.log(this.name); // undefined
  },
  // 正确示例
  greetRight() {
    console.log(this.name); // "张三"
  }
};
```

### 7.5 类（Class）

```javascript
class Person {
  // 构造函数
  constructor(name, age) {
    this.name = name;
    this.age = age;
  }

  // 实例方法
  greet() {
    console.log(`我是 ${this.name}`);
  }

  // 静态方法
  static createAnonymous() {
    return new Person("匿名", 0);
  }

  // Getter/Setter
  get info() {
    return `${this.name} ${this.age}岁`;
  }
  set age(value) {
    if (value < 0) throw new Error("年龄不能为负");
    this._age = value;
  }
}

// 继承
class Student extends Person {
  constructor(name, age, grade) {
    super(name, age);
    this.grade = grade;
  }

  study() {
    console.log(`${this.name} 正在学习`);
  }
}
```

### 7.6 模块化

```javascript
// 导出（math.js）
export const PI = 3.14;
export function add(a, b) {
  return a + b;
}
export default function multiply(a, b) {
  return a * b;
}

// 导入
import multiply, { add, PI } from "./math.js";
import * as math from "./math.js";
```

### 7.7 Promise 与 Async/Await

```javascript
// Promise
function fetchData() {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      resolve("数据加载成功");
    }, 1000);
  });
}

fetchData()
  .then(data => console.log(data))
  .catch(error => console.error(error))
  .finally(() => console.log("完成"));

// Async/Await
async function loadData() {
  try {
    const data = await fetchData();
    console.log(data);
  } catch (error) {
    console.error(error);
  } finally {
    console.log("完成");
  }
}
```

## 八、JavaScript 最佳实践

### 8.1 代码规范

```javascript
// 使用 const/let，避免 var
const MAX_SIZE = 100;
let currentSize = 0;

// 使用 === 和 !==，避免 == 和 !=
if (x === "5") { /* ... */ } // 推荐
if (x == "5") { /* ... */ }  // 不推荐

// 使用有意义的变量名
const userAge = 25;  // 好
const a = 25;        // 差

// 函数单一职责
function calculateArea(length, width) { /* ... */ } // 好
function process(data, options, callback) { /* ... */ } // 差
```

### 8.2 错误处理

```javascript
// try-catch
try {
  const data = JSON.parse(jsonString);
} catch (error) {
  console.error("解析失败:", error.message);
} finally {
  console.log("清理资源");
}

// 抛出错误
function divide(a, b) {
  if (b === 0) {
    throw new Error("除数不能为零");
  }
  return a / b;
}
```

### 8.3 防抖与节流

```javascript
// 防抖：延迟执行，频繁触发时重置计时
function debounce(func, delay) {
  let timer;
  return function(...args) {
    clearTimeout(timer);
    timer = setTimeout(() => func.apply(this, args), delay);
  };
}

// 节流：固定时间间隔执行
function throttle(func, delay) {
  let lastCall = 0;
  return function(...args) {
    const now = Date.now();
    if (now - lastCall >= delay) {
      lastCall = now;
      func.apply(this, args);
    }
  };
}

// 使用
const handleSearch = debounce(function(value) {
  console.log("搜索:", value);
}, 300);

const handleScroll = throttle(function() {
  console.log("滚动中...");
}, 200);
```

## 九、学习检查清单

掌握以下知识点表示你已打好 JavaScript 基础：

- [ ] JavaScript 三种导入方式
- [ ] var、let、const 的区别
- [ ] 基本数据类型和引用类型
- [ ] 类型转换和类型检测
- [ ] 条件语句（if, switch, 三元运算符）
- [ ] 循环语句（for, while, for...in, for...of）
- [ ] 函数定义方式（声明、表达式、箭头函数）
- [ ] 函数参数（默认参数、剩余参数、解构参数）
- [ ] 作用域和闭包
- [ ] this 关键字
- [ ] 对象和数组的基础操作
- [ ] 数组常用方法（map, filter, reduce 等）
- [ ] DOM 元素获取
- [ ] DOM 内容和样式修改
- [ ] DOM 元素创建和删除
- [ ] 事件绑定和处理
- [ ] 事件对象和事件传播
- [ ] ES6+ 新特性（模板字符串、解构、类、Promise 等）

## 总结

JavaScript 是网页的灵魂，掌握其语法和 DOM 操作是实现网页交互的关键。通过本文的学习，你应该能够：

1. 使用变量、数据类型和运算符
2. 编写函数和理解作用域
3. 操作 DOM 改变页面内容和样式
4. 处理用户交互事件
5. 使用 ES6+ 现代语法

下一步建议深入学习 JavaScript 高级特性，或学习前端框架（React/Vue）。
