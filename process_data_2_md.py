import json

def summarize_to_markdown(json_file, output_file="summary.md"):
    """
    从 JSON 文件中提取实验参数和最佳准确率，生成 Markdown 表格。
    Args:
        json_file (str): 输入的 JSON 文件路径。
        output_file (str): 输出的 Markdown 文件路径。
    """
    try:
        # 读取 JSON 文件
        with open(json_file, "r") as f:
            data = json.load(f)
        
        # 初始化 Markdown 表格
        markdown = "| Experiment   | Image Size | Patch Size | Dim  | Depth | Heads | MLP Dim | Accuracy |\n"
        markdown += "|--------------|------------|------------|------|-------|-------|---------|----------|\n"
        
        # 遍历每个实验
        for experiment, details in data.items():
            params = details.get("params", {})
            best_accuracy = details.get("best_accuracy", {})
            
            # 提取参数
            image_size = params.get("image_size", "N/A")
            patch_size = params.get("patch_size", "N/A")
            dim = params.get("dim", "N/A")
            depth = params.get("depth", "N/A")
            heads = params.get("heads", "N/A")
            mlp_dim = params.get("mlp_dim", "N/A")
            accuracy = best_accuracy.get("accuracy", "N/A")
            
            # 添加到表格
            markdown += f"| {experiment} | {image_size}         | {patch_size}         | {dim}  | {depth}     | {heads}     | {mlp_dim}    | {accuracy:.4f} |\n"
        
        # 写入 Markdown 文件
        with open(output_file, "w") as f:
            f.write(markdown)
        
        print(f"Markdown 表格已保存到 {output_file}")
    
    except Exception as e:
        print(f"处理文件时出错: {e}")

# 示例调用
json_file = "vit_params_summary.json"  # 替换为实际路径
summarize_to_markdown(json_file)