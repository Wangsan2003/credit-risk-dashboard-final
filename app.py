import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import dash
from dash import dcc, html

# --- 1. 数据准备与模型训练 (重现模块 1-6 的商业价值输出) ---

# 1.1 加载数据与特征工程 (模块 1-4 核心)
try:
    # 假设 UCI_Credit_Card.csv 在同一目录下
    df = pd.read_csv('UCI_Credit_Card.csv')
except FileNotFoundError:
    raise FileNotFoundError("错误：UCI_Credit_Card.csv 文件未找到。请确保它与 app.py 在同一目录下。")

df = df.rename(columns={'default.payment.next.month': 'DEFAULT', 'PAY_0': 'PAY_1'})
df = df.drop('ID', axis=1)
df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})

bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
pay_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
pay_status_cols = [f'PAY_{i}' for i in range(1, 7)]

# 模块 4: 关键衍生特征
df['AVG_UTILIZATION'] = df[bill_cols].mean(axis=1) / df['LIMIT_BAL']
df['AVG_UTILIZATION'] = df['AVG_UTILIZATION'].replace([np.inf, -np.inf], 0).clip(upper=5)
df['TOTAL_DEBT_BALANCE'] = df[bill_cols].sum(axis=1) - df[pay_cols].sum(axis=1)
df['MAX_DELAY'] = df[pay_status_cols].apply(max, axis=1).clip(lower=-2)
df['LATE_COUNT'] = (df[pay_status_cols] >= 1).sum(axis=1)

target = df['DEFAULT']
features = df.drop('DEFAULT', axis=1)
categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
numerical_cols = [col for col in features.columns if col not in categorical_cols]

features_cat_encoded = pd.get_dummies(features[categorical_cols], drop_first=False)
scaler = StandardScaler()
features_num_scaled = pd.DataFrame(
    scaler.fit_transform(features[numerical_cols]),
    columns=numerical_cols
)
df_featured_final = pd.concat([features_num_scaled, features_cat_encoded, target], axis=1)

X = df_featured_final.drop('DEFAULT', axis=1)
y = df_featured_final['DEFAULT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 1.2 模型训练 (模块 5 核心)
ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic', use_label_encoder=False, eval_metric='logloss',
    scale_pos_weight=ratio, random_state=42, n_estimators=200, max_depth=5
)
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
xgb_model.fit(X_train, y_train)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, xgb_probs)

# 1.3 模型解释结果 (模块 6 核心)
feature_importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
top_10_xgb = feature_importances.nlargest(10).sort_values(ascending=True)


# --- 2. Plotly 图表生成函数 ---

def create_roc_pr_plots(y_true, y_probs, roc_auc):
    """模块 5: 模型评估的可视化"""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)
    no_skill = y_true.sum() / len(y_true)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC 曲线 (AUC = {:.4f})".format(roc_auc),
                                                        "精确率-召回率曲线 (AP = {:.4f})".format(avg_precision)))

    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC', line=dict(color='#ff6347')), row=1, col=1)
    fig.add_trace(go.Line(x=[0, 1], y=[0, 1], name='随机猜测', line=dict(dash='dash', color='#3498db')), row=1, col=1)

    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='PR', line=dict(color='#2ecc71')), row=1, col=2)
    fig.add_trace(go.Line(x=[0, 1], y=[no_skill, no_skill], name=f'随机基线 ({no_skill:.2f})',
                          line=dict(dash='dash', color='#9b59b6')), row=1, col=2)

    fig.update_layout(height=450, margin={'l': 20, 'r': 20, 't': 50, 'b': 20}, showlegend=True,
                      plot_bgcolor='#fcfcfc', paper_bgcolor='#f8f9fa')
    fig.update_xaxes(title_text="假阳性率 / 召回率", row=1, col=1)
    fig.update_xaxes(title_text="召回率", row=1, col=2)
    return fig


def create_risk_profile_plot(df_original):
    """模块 3/4: 违约用户画像的可视化 (LIMIT_BAL 和 EDUCATION)"""
    df_plot = df_original.copy()
    df_plot['DEFAULT_STATUS'] = df_plot['DEFAULT'].map({0: '未违约', 1: '已违约'})

    fig = make_subplots(rows=1, cols=2, subplot_titles=("信用额度 (LIMIT_BAL) 分布", "教育程度 (EDUCATION) 违约率"))

    # 图表 1: LIMIT_BAL Box Plot (模块 3)
    box_fig = px.box(df_plot, x='DEFAULT_STATUS', y='LIMIT_BAL',
                     color='DEFAULT_STATUS',
                     color_discrete_map={'未违约': '#3498db', '已违约': '#e74c3c'})
    fig.add_trace(box_fig.data[0], row=1, col=1)
    fig.add_trace(box_fig.data[1], row=1, col=1)

    # 图表 2: Education Bar Plot (模块 3)
    edu_rate = df_plot.groupby('EDUCATION')['DEFAULT'].mean().reset_index()
    bar_fig = px.bar(edu_rate, x='EDUCATION', y='DEFAULT',
                     color='DEFAULT',
                     color_continuous_scale='Reds',
                     labels={'DEFAULT': '违约率'})
    fig.add_trace(bar_fig.data[0], row=1, col=2)

    fig.update_layout(height=450, title_text="模块 II：关键用户画像与特征分布", showlegend=False,
                      margin={'l': 20, 'r': 20, 't': 50, 'b': 20},
                      plot_bgcolor='#fcfcfc', paper_bgcolor='#f8f9fa')
    fig.update_yaxes(title_text="信用额度 (NTD)", range=[0, 500000], row=1, col=1)
    fig.update_yaxes(title_text="违约率", row=1, col=2)
    fig.update_xaxes(title_text="违约状态", row=1, col=1)
    fig.update_xaxes(title_text="教育程度 (1:研究生, 2:大学, 3:高中/其他)", row=1, col=2)

    return fig


def create_feature_importance_plot(importance_series):
    """模块 6: 风险解释与规则的可视化"""
    fig = px.bar(importance_series, orientation='h',
                 labels={'value': '重要性得分 (F-Score)', 'index': '特征名称'},
                 color=importance_series.index,
                 color_discrete_sequence=px.colors.qualitative.D3)
    fig.update_layout(showlegend=False, height=450,
                      yaxis={'categoryorder': 'total ascending'},
                      title='模块 I：XGBoost 特征重要性 (风险驱动因素)',
                      margin={'l': 20, 'r': 20, 't': 50, 'b': 20},
                      plot_bgcolor='#fcfcfc', paper_bgcolor='#f8f9fa')
    return fig


# --- 3. Dash 应用程序布局 (集成六个模块的商业价值) ---

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server  # 必须保留，用于 Render 部署

app.layout = html.Div(style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'fontFamily': 'Arial, sans-serif'},
                      children=[

                          html.H1("信用卡违约风险分析仪表盘 (最终交付)",
                                  style={'textAlign': 'center', 'color': '#2c3e50'}),
                          html.P(f"最佳预测模型：XGBoost | 性能：测试集 AUC-ROC: {roc_auc:.4f}",
                                 style={'textAlign': 'center', 'color': '#34495e', 'marginBottom': '40px'}),

                          # 模块 I & VI: 风险驱动因素 (解释与规则)
                          html.Div(className='row',
                                   style={'marginBottom': '30px', 'backgroundColor': 'white', 'padding': '20px',
                                          'borderRadius': '8px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.1)'}, children=[
                                  html.Div(className='six columns', children=[
                                      html.H3("模块 I: 风险解释与驱动因素", style={'color': '#2980b9'}),
                                      html.P("【模块 6 价值】展示模型决策依据和最高风险特征。",
                                             style={'color': '#7f8c8d'}),
                                      dcc.Graph(figure=create_feature_importance_plot(top_10_xgb))
                                  ]),
                                  html.Div(className='six columns', style={'paddingTop': '20px'}, children=[
                                      html.H3("关键业务行动建议 (基于模块 6)", style={'color': '#e74c3c'}),
                                      html.Ul(
                                          style={'paddingLeft': '20px', 'lineHeight': '1.8', 'listStyleType': 'none'},
                                          children=[
                                              html.Li(html.Strong("1. 最高优先级："),
                                                      "将 MAX_DELAY (最大延迟月数) 设为信审和贷后管理的最高权重变量。"),
                                              html.Li(html.Strong("2. 量化规则："),
                                                      "LATE_COUNT (延迟次数) 的风险倍率达 2.18 倍，应根据该指标触发强力预警。"),
                                              html.Li(html.Strong("3. 财务监控："),
                                                      "对低 LIMIT_BAL 和高 AVG_UTILIZATION (平均使用率) 的客户进行重点风险关注。"),
                                              html.Li(html.Strong("4. 模型应用："),
                                                      "使用 XGBoost 进行自动化评分，但用 LR 的系数进行业务规则解释和校准。"),
                                          ])
                                  ])
                              ]),

                          # 模块 II & V: 模型性能监控
                          html.Div(className='row',
                                   style={'marginBottom': '30px', 'backgroundColor': 'white', 'padding': '20px',
                                          'borderRadius': '8px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.1)'}, children=[
                                  html.H3("模块 II: XGBoost 模型性能评估 (模块 5)",
                                          style={'textAlign': 'center', 'color': '#2980b9'}),
                                  dcc.Graph(figure=create_roc_pr_plots(y_test, xgb_probs, roc_auc))
                              ]),

                          # 模块 III & IV: 违约用户画像
                          html.Div(className='row',
                                   style={'marginBottom': '30px', 'backgroundColor': 'white', 'padding': '20px',
                                          'borderRadius': '8px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.1)'}, children=[
                                  html.H3("模块 III: 违约用户画像与特征分布 (模块 3/4)",
                                          style={'textAlign': 'center', 'color': '#2980b9'}),
                                  dcc.Graph(figure=create_risk_profile_plot(df))
                              ])
                      ])

if __name__ == '__main__':
    # 在本地运行，需要 debug=True
    app.run(debug=True)         # <-- 修正为 app.run()