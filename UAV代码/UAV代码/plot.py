import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# 设置中文字体（增加多个备选，确保兼容不同环境）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong', 'KaiTi', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10  # 全局基础字号提升

fig, ax = plt.subplots(1, 1, figsize=(18, 24))  # 画布放大，提升整体显示空间
ax.set_xlim(0, 18)
ax.set_ylim(0, 24)
ax.axis('off')

# 定义颜色方案（保持原风格，优化对比度）
colors = {
    'title': '#E8B87D',  # 橙色标题框
    'main': '#A8D5BA',  # 浅绿色主框
    'sub': '#C8E6C9',  # 更浅绿色子框
    'highlight': '#90CAF9',  # 蓝色高亮
    'border': '#333333',  # 边框颜色
    'arrow': '#1976D2',  # 箭头颜色
    'side': '#5C6BC0',  # 侧边栏颜色
    'side_text': 'white'  # 侧边栏文字
}

# 绘制侧边栏阶段标签（字号放大，间距调整）
stages = [
    ('数据获取', 22.5, 21.5),
    ('数据预处理', 21, 19),
    ('特征工程', 18.5, 16.5),
    ('特征筛选', 16, 14),
    ('差异化建模', 13.5, 8.5),
    ('结果分析', 8, 6.5),
    ('结论展望', 6, 3.5)
]

for i, (name, y_top, y_bottom) in enumerate(stages):
    # 绘制侧边栏（加宽高度，提升视觉占比）
    rect = FancyBboxPatch((0.5, y_bottom), 1.6, y_top - y_bottom,
                          boxstyle="round,pad=0.05",
                          facecolor=colors['side'],
                          edgecolor=colors['border'],
                          linewidth=2.5)
    ax.add_patch(rect)
    # 添加文字（字号放大到13，加粗增强可读性）
    ax.text(1.3, (y_top + y_bottom) / 2, name,
            ha='center', va='center',
            fontsize=13, fontweight='bold',
            color=colors['side_text'],
            rotation=90)

# ==================== 第1阶段：数据获取 ====================
# 标题框（放大尺寸）
title1 = FancyBboxPatch((2.5, 21.5), 15, 1.2,
                        boxstyle="round,pad=0.08",
                        facecolor=colors['title'],
                        edgecolor=colors['border'],
                        linewidth=2.5)
ax.add_patch(title1)
ax.text(10, 22.1, '无人机高光谱数据获取与实测数据采集',
        ha='center', va='center', fontsize=14, fontweight='bold')

# 子框（放大尺寸，字号提升）
sub1_items = [
    '无人机高光谱影像获取',
    '同步水质实测数据采集',
    '时空匹配数据构建'
]
for i, item in enumerate(sub1_items):
    x = 3 + i * 4.8
    box = FancyBboxPatch((x, 21.8), 4.2, 0.8,
                         boxstyle="round,pad=0.05",
                         facecolor=colors['main'],
                         edgecolor=colors['border'],
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x + 2.1, 22.2, item, ha='center', va='center', fontsize=12, fontweight='bold')

# ==================== 第2阶段：数据预处理 ====================
title2 = FancyBboxPatch((2.5, 19), 15, 1.2,
                        boxstyle="round,pad=0.08",
                        facecolor=colors['title'],
                        edgecolor=colors['border'],
                        linewidth=2.5)
ax.add_patch(title2)
ax.text(10, 19.6, '高光谱数据预处理与水体信息提取',
        ha='center', va='center', fontsize=14, fontweight='bold')

# 子框（优化间距，字号放大到11）
sub2_items = [
    ('几何校正', 3.5),
    ('辐射校正', 6.5),
    ('反射率转换', 9.5),
    ('NDWI水体提取', 12.5),
    ('光谱标准化', 15.5)
]
for text, x in sub2_items:
    box = FancyBboxPatch((x, 19.3), 2.5, 0.7,
                         boxstyle="round,pad=0.05",
                         facecolor=colors['main'],
                         edgecolor=colors['border'],
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x + 1.25, 19.65, text, ha='center', va='center', fontsize=11, fontweight='bold')

# ==================== 第3阶段：特征工程 ====================
title3 = FancyBboxPatch((2.5, 16.5), 15, 1.2,
                        boxstyle="round,pad=0.08",
                        facecolor=colors['title'],
                        edgecolor=colors['border'],
                        linewidth=2.5)
ax.add_patch(title3)
ax.text(10, 17.1, '多源光谱特征构建',
        ha='center', va='center', fontsize=14, fontweight='bold')

# 三大特征类型（放大尺寸，字号提升到12）
feature_types = [
    ('原始光谱特征', 4, '#FFCCBC'),
    ('光谱指数特征', 8.5, '#C5CAE9'),
    ('导数光谱特征', 13, '#B2DFDB')
]
for name, x, color in feature_types:
    # 主框
    box = FancyBboxPatch((x, 16.8), 4, 0.8,
                         boxstyle="round,pad=0.05",
                         facecolor=color,
                         edgecolor=colors['border'],
                         linewidth=2.5)
    ax.add_patch(box)
    ax.text(x + 2, 17.2, name, ha='center', va='center',
            fontsize=12, fontweight='bold')

# ==================== 第4阶段：特征筛选 ====================
title4 = FancyBboxPatch((2.5, 14), 15, 1.2,
                        boxstyle="round,pad=0.08",
                        facecolor=colors['title'],
                        edgecolor=colors['border'],
                        linewidth=2.5)
ax.add_patch(title4)
ax.text(10, 14.6, '关键特征筛选（Spearman + SHAP）',
        ha='center', va='center', fontsize=14, fontweight='bold')

# 两阶段筛选（字号放大到11，优化布局）
stage4_boxes = [
    ('Spearman相关性分析\n初步筛选', 5, '#FFE0B2'),
    ('SHAP特征重要性评估\n精准提取', 12, '#DCEDC8')
]
for text, x, color in stage4_boxes:
    box = FancyBboxPatch((x, 14.3), 4, 0.8,
                         boxstyle="round,pad=0.05",
                         facecolor=color,
                         edgecolor=colors['border'],
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x + 2, 14.7, text, ha='center', va='center', fontsize=11, fontweight='bold')

# 箭头连接（加粗箭头）
arrow = FancyArrowPatch((9, 14.7), (11, 14.7),
                        arrowstyle='->', mutation_scale=25,
                        linewidth=3, color=colors['arrow'])
ax.add_patch(arrow)
ax.text(10, 15.05, '关键特征子集', ha='center', va='center',
        fontsize=11, style='italic', color=colors['arrow'], fontweight='bold')

# ==================== 第5阶段：差异化建模（核心） ====================
title5 = FancyBboxPatch((2.5, 12.5), 15, 1,
                        boxstyle="round,pad=0.08",
                        facecolor='#FF8A65',
                        edgecolor=colors['border'],
                        linewidth=2.5)
ax.add_patch(title5)
ax.text(10, 13, '水质参数差异化反演建模',
        ha='center', va='center', fontsize=14, fontweight='bold', color='white')

# 三个参数的分支（放大尺寸，字号提升）
params = [
    {
        'name': '溶解氧（DO）',
        'color': '#FFCCBC',
        'y': 11.5,
        'methods': [
            '混合专家模型MoE',
            'CatBoost+XGBoost+\nRF+LightGBM',
            '双门控机制',
            '自适应数据增强'
        ]
    },
    {
        'name': '总溶解性固体（TDS）',
        'color': '#C5CAE9',
        'y': 9,
        'methods': [
            'CatBoost回归模型',
            'SNV+SG导数变换',
            'SHAP关键波段筛选',
            '非线性关系建模'
        ]
    },
    {
        'name': '浊度（TURB）',
        'color': '#B2DFDB',
        'y': 6.5,
        'methods': [
            '分期XGBoost建模',
            '两阶段特征筛选',
            '多时相分别训练',
            '时空异质性适应'
        ]
    }
]

for param in params:
    # 参数标题（放大尺寸）
    title_box = FancyBboxPatch((2.8, param['y'] + 0.5), 3.5, 0.8,
                               boxstyle="round,pad=0.05",
                               facecolor=param['color'],
                               edgecolor=colors['border'],
                               linewidth=2.5)
    ax.add_patch(title_box)
    ax.text(4.55, param['y'] + 0.9, param['name'],
            ha='center', va='center', fontsize=12, fontweight='bold')

    # 方法框（放大尺寸，字号提升到10）
    for i, method in enumerate(param['methods']):
        x = 7 + i * 2.6
        method_box = FancyBboxPatch((x, param['y'] + 0.3), 2.4, 1,
                                    boxstyle="round,pad=0.03",
                                    facecolor=colors['sub'],
                                    edgecolor=colors['border'],
                                    linewidth=1.5)
        ax.add_patch(method_box)
        ax.text(x + 1.2, param['y'] + 0.8, method,
                ha='center', va='center', fontsize=10, fontweight='bold')

# ==================== 第6阶段：结果分析 ====================
title6 = FancyBboxPatch((2.5, 6.5), 15, 1.2,
                        boxstyle="round,pad=0.08",
                        facecolor=colors['title'],
                        edgecolor=colors['border'],
                        linewidth=2.5)
ax.add_patch(title6)
ax.text(10, 7.1, '水质参数反演结果分析与验证',
        ha='center', va='center', fontsize=14, fontweight='bold')

# 分析内容（字号放大到11，优化布局）
analysis_items = [
    ('精度评价\n(R²/RMSE/MAE)', 4, '#FFF9C4'),
    ('空间反演制图', 7.8, '#F8BBD9'),
    ('时空分布特征分析', 11.6, '#E1BEE7'),
    ('方法验证与对比', 15.4, '#B3E5FC')
]
for text, x, color in analysis_items:
    box = FancyBboxPatch((x, 6.8), 3, 0.7,
                         boxstyle="round,pad=0.05",
                         facecolor=color,
                         edgecolor=colors['border'],
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x + 1.5, 7.15, text, ha='center', va='center', fontsize=11, fontweight='bold')

# ==================== 第7阶段：结论展望 ====================
title7 = FancyBboxPatch((2.5, 3.5), 15, 1.2,
                        boxstyle="round,pad=0.08",
                        facecolor=colors['title'],
                        edgecolor=colors['border'],
                        linewidth=2.5)
ax.add_patch(title7)
ax.text(10, 4.1, '结论与展望',
        ha='center', va='center', fontsize=14, fontweight='bold')

conclusion_items = [
    '主要研究结论',
    '方法创新点总结',
    '研究不足与改进方向',
    '未来展望'
]
for i, item in enumerate(conclusion_items):
    x = 3.5 + i * 3.6
    box = FancyBboxPatch((x, 3.8), 3, 0.7,
                         boxstyle="round,pad=0.05",
                         facecolor=colors['main'],
                         edgecolor=colors['border'],
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x + 1.5, 4.15, item, ha='center', va='center', fontsize=11, fontweight='bold')

# ==================== 绘制连接箭头（加粗放大） ====================
# 主流程箭头
main_arrows = [
    (22.1, 21.5),
    (19.6, 19),
    (17.1, 16.5),
    (14.6, 14),
    (13, 8.5),
    (7.1, 6.5),
    (4.1, 3.5)
]

for y_start, y_end in main_arrows:
    arrow = FancyArrowPatch((10, y_start), (10, y_end + 0.2),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=4, color=colors['arrow'])
    ax.add_patch(arrow)

# ==================== 添加虚线框（加粗） ====================
dashed_boxes = [
    (2.2, 21.3, 15.6, 1.8, '数据层'),
    (2.2, 18.8, 15.6, 1.8, '预处理层'),
    (2.2, 16.3, 15.6, 1.8, '特征层'),
    (2.2, 13.8, 15.6, 1.8, '筛选层'),
    (2.2, 6.3, 15.6, 6.5, '模型层'),
    (2.2, 6.3, 15.6, 1.8, '分析层'),
    (2.2, 3.3, 15.6, 1.8, '总结层')
]

for x, y, w, h, label in dashed_boxes:
    rect = Rectangle((x, y), w, h, linewidth=2,
                     edgecolor='gray', facecolor='none',
                     linestyle='--', alpha=0.7)
    ax.add_patch(rect)

# ==================== 添加图例说明（字号放大） ====================
legend_y = 1.5
ax.text(10, legend_y + 0.5, '图例说明', ha='center', va='center',
        fontsize=13, fontweight='bold')

legend_items = [
    ('数据获取与预处理', colors['title']),
    ('特征工程与筛选', colors['main']),
    ('DO反演', '#FFCCBC'),
    ('TDS反演', '#C5CAE9'),
    ('TURB反演', '#B2DFDB')
]

for i, (name, color) in enumerate(legend_items):
    x = 3 + i * 3.2
    rect = Rectangle((x, legend_y - 0.2), 0.6, 0.4,
                     facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + 0.8, legend_y, name, ha='left', va='center', fontsize=10, fontweight='bold')

# ==================== 标题（放大字号，优化间距） ====================
ax.text(10, 23.8, '基于无人机高光谱遥感的淡水养殖区水质参数反演方法研究技术路线',
        ha='center', va='center', fontsize=16, fontweight='bold')
ax.text(10, 23.3, 'Technical Roadmap of Water Quality Parameter Retrieval in Freshwater Aquaculture Areas '
                 'Based on UAV Hyperspectral Remote Sensing',
        ha='center', va='center', fontsize=12, style='italic')

plt.tight_layout()
# 保存时优化分辨率和边距
plt.savefig('technical_roadmap_water_quality_v2.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('technical_roadmap_water_quality_v2.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("优化后的技术路线图已保存: technical_roadmap_water_quality_v2.png 和 .pdf")
plt.show()