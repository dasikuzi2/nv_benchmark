# 内存带宽测试

## 运行

```bash
make run
```

## 测试内容

1. **Block配置**：测试并发度影响（1x到8x SMs）
2. **L2缓存**：测试数据大小影响（10MB到1GB）
3. **向量化**：对比float和float4性能

## 预期结果 (A800)

```
Block配置: 8x SMs → 1236 GB/s
L2缓存: <40MB → 1700+ GB/s
向量化: float4 → 1723 GB/s (+125%)
```

## 核心代码

```c
// 向量化加载（关键优化）
float4 val = input[i];  // 128-bit load
sum.x += val.x;
sum.y += val.y;
sum.z += val.z;
sum.w += val.w;
```

## 常见问题

**Q: 带宽低于预期？**
- 确保使用 8x SM blocks
- 使用 float4 向量化
- 检查内存访问是否合并
