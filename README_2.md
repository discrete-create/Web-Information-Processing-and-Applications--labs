# Part3
## 倒排表扩展与优化
- 输入: 基础倒排索引(`inverted_index.json`) + 规范化词表(`normalized_tokens.json`)
- 输出: 增强倒排索引 (`enhanced_inverted_index.json`)
- 输出格式见下
- 添加功能:
  - 位置信息添加（为后续短语检索做准备）
  - 按块存储压缩（块大小=4）
  - 前端编码压缩（共同前缀利用）
  - 压缩效果分析与比较

## 接口

### 主要类: `EnhancedInvertedIndex`

#### 初始化
```python
index = EnhancedInvertedIndex()
```

#### 数据加载
```python
index.load_from_basic_index("inverted_index.json", "normalized_tokens.json")
```

#### 压缩实现
```python
index.implement_blocking_compression()      # 按块存储
index.implement_front_coding_compression()  # 前端编码
```


#### 统计信息
```python
index.print_compression_statistics()  # 打印压缩统计
```

#### 保存结果
```python
index.save_enhanced_index("enhanced_inverted_index.json")
```

## 数据结构

### 倒排记录格式
```python
[文档ID, 词频, 位置列表, 跳表指针]
# 示例: ["Group-123", 2, [3, 7], -1]
```

### 压缩块格式
```python
{
    "terms": ["词项1", "词项2", "词项3", "词项4"],
    "term_ids": [20, 21, 22, 23],
    "postings": [[词项ID, [文档ID, 词频, [位置列表], 跳表指针]], ...]
}
```

### 前端编码格式
```python
{
    "type": "prefix",
    "prefix": "共同前缀",
    "suffixes": ["后缀1", "后缀2", "后缀3"]
}
```

### 增强倒排索引结构
```json
{
  "vocabulary": {"词项": 词项ID},
  "inverted_lists": {
    "词项ID": [["文档ID", 词频, [位置列表], 跳表指针]]
  },
  "doc_lengths": {"文档ID": 文档长度},
  "doc_sort_keys": {"文档ID": 排序键},
  "sort_key_to_doc": {"排序键": "文档ID"},
  "next_sort_key": 下一个排序键,
  "has_positions": true,
  "blocking_compression": {
    "compressed_vocabulary": {"词项": {"block_id": 块ID, "local_id": 局部ID}},
    "compressed_lists": {
      "块ID": {
        "terms": ["词项1", "词项2", ...],
        "term_ids": [词项ID1, 词项ID2, ...],
        "postings": [[词项ID, [文档ID, 词频, [位置列表], 跳表指针]], ...]
      }
    },
    "block_size": 4
  },
  "front_coding_compression": {
    "front_coded_vocabulary": {"词项": {"entry_index": 条目索引, "suffix_index": 后缀索引}},
    "front_coded_terms": [
      {"type": "prefix", "prefix": "共同前缀", "suffixes": ["后缀1", "后缀2"]},
      {"type": "single", "term": "单个词项"}
    ]
  },
  "compression_methods": ["blocking", "front_coding"]
}
```
  

