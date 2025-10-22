# Part4: 多种形式的信息检索

## 项目概述
Part4实现了多种形式的信息检索系统，包括布尔检索和向量空间模型，并进行全面的性能分析。

## 核心功能
- ✅ 布尔检索：支持复杂布尔表达式查询
- ✅ 短语检索：基于位置信息的精确匹配
- ✅ 向量空间模型：TF-IDF计算和余弦相似度
- ✅ 性能分析：查询优化、压缩效果、跳表指针影响
- ✅ 查询设计：3种复杂布尔查询条件

## 接口

### 主要类: `InformationRetrievalSystem`

#### 初始化
```python
retrieval_system = InformationRetrievalSystem(
    "enhanced_inverted_index.json",
    "normalized_tokens.json"
)
```

#### 布尔检索
```python
result = retrieval_system.boolean_retrieval("web & development")
# 返回: 匹配的文档集合
```

#### 短语检索
```python
result = retrieval_system.phrase_retrieval(["web", "development"])
# 返回: 匹配的文档集合
```

#### 向量空间检索
```python
results = retrieval_system.vector_space_retrieval(
    ["web", "development", "technology"], 
    top_k=5
)
# 返回: [(文档ID, 相似度分数), ...]
```

#### 性能分析
```python
# 查询处理顺序分析
timing_results = retrieval_system.analyze_query_processing_order(query)

# 压缩效果比较
compression_results = retrieval_system.compare_compression_efficiency(query)

# 跳表指针分析
skip_results = retrieval_system.analyze_skip_pointer_impact(query, [1, 2, 4, 8])
```

#### TF-IDF计算
```python
tf_idf = retrieval_system.calculate_tf_idf("Group-123", "web")
# 返回: TF-IDF值
```

## 数据结构

### 布尔查询格式
```
查询1: "web & development"           # 简单AND查询
查询2: "machine | learning"          # 简单OR查询
查询3: "tech & (enthusiasts | conference)"  # 复杂组合查询
```

### 短语查询格式
```python
phrase_queries = [
    ["web", "development"],
    ["machine", "learning"],
    ["ai", "conference"],
    ["tech", "enthusiasts"]
]
```

### 向量查询格式
```python
vector_queries = [
    ["web", "development", "technology"],
    ["machine", "learning", "ai"],
    ["data", "science", "analysis"],
    ["tech", "enthusiasts", "conference"]
]
```

### 检索结果格式
```python
# 布尔检索结果
result = {"Group-123", "PastEvent-789"}

# 向量空间检索结果
results = [
    ("Group-123", 0.8542),
    ("PastEvent-789", 0.7231),
    ("Memeber-456", 0.6890)
]
```

## 核心算法

### 1. 布尔表达式解析
```python
def _parse_boolean_query(self, query: str) -> List[str]:
    # 词法分析，将查询字符串分解为词项和操作符
```

### 2. 中缀转后缀表达式
```python
def _infix_to_postfix(self, tokens: List[str]) -> List[str]:
    # 使用栈将中缀表达式转换为后缀表达式
```

### 3. 后缀表达式求值
```python
def _evaluate_postfix(self, postfix: List[str]) -> Set[str]:
    # 使用栈计算后缀表达式的结果
```

### 4. 短语匹配算法
```python
def _check_phrase_in_doc(self, doc_id: str, phrase: List[str], start_positions: List[int]) -> bool:
    # 基于位置信息检查短语是否在文档中连续出现
```

### 5. TF-IDF计算
```python
def calculate_tf_idf(self, doc_id: str, term: str) -> float:
    # TF = 词频 / 文档长度
    # IDF = log(文档总数 / 包含该词的文档数)
    # TF-IDF = TF * IDF
```

### 6. 余弦相似度计算
```python
def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    # 余弦相似度 = 向量点积 / (向量模长1 * 向量模长2)
```

## 性能分析

### 查询处理顺序优化
- **原始顺序**: 按查询表达式顺序处理
- **优化顺序**: 按词项频率排序，高频词项优先处理
- **效果**: 减少中间结果集大小，提升处理速度

### 压缩效果分析
- **原始索引**: 使用基础倒排索引
- **压缩索引**: 使用按块存储和前端编码压缩的索引
- **指标**: 检索时间、存储空间、结果一致性

### 跳表指针步长分析
- **测试步长**: [1, 2, 4, 8]
- **分析指标**: 检索时间、存储开销
- **最优步长**: 根据数据集特点确定

### 位置信息效果
- **短语检索**: 基于词项位置信息的精确匹配
- **效果分析**: 对比有无位置信息的检索准确性
- **应用场景**: 精确短语查询、邻近词查询

## 运行方式

### 方法1: 运行完整系统
```python
exec(open('Part4_Information_Retrieval.py').read())
```

### 方法2: 分步骤运行（推荐）
在Jupyter Notebook中按顺序运行各个cell，可以观察每个功能的详细过程。

## 输入输出

### 输入文件
- `enhanced_inverted_index.json` - 增强倒排索引（来自Part3）
- `normalized_tokens.json` - 规范化词表（来自Part1）

### 输出结果
- 布尔检索结果：匹配的文档集合
- 短语检索结果：精确匹配的文档集合
- 向量空间检索结果：按相似度排序的文档列表
- 性能分析报告：各种优化策略的效果对比

### 控制台输出
- 各检索方法的执行过程
- 性能分析结果
- 检索效果统计
- 系统特点总结
