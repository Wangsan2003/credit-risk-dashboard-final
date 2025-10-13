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

# 1.1 数据加载与特征工程 (模块 1-4 核心)
try:
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

# 1.4 逻辑回归模型 (用于 Odds Ratio 文本提示)
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
lr_model = LogisticRegression(solver='liblinear', random_state=42, C=0.01)
lr_model.fit(X_smote, y_smote)
lr_coefficients = pd.Series(lr_model.coef_[0], index=X.columns)
late_count_odds = np.exp(lr_coefficients['LATE_COUNT']).round(2)


# --- 2. Plotly 图表生成函数 ---

def create_roc_pr_plots(y_true, y_probs, roc_auc):
    """模块 I: 模型性能 (ROC & PR)"""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)
    no_skill = y_true.sum() / len(y_true)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC 曲线 (AUC = {:.4f})".format(roc_auc),
                                                        "精确率-召回率曲线 (AP = {:.4f})".format(avg_precision)))

    # 使用 go.Scatter 替代 go.Line 来避免 DeprecationWarning
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC', line=dict(color='#e74c3c', width=2)), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='随机猜测', line=dict(dash='dash', color='#3498db', width=1)),
        row=1, col=1)

    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='PR', line=dict(color='#2ecc71', width=2)),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[no_skill, no_skill], mode='lines', name=f'随机基线 ({no_skill:.2f})',
                             line=dict(dash='dash', color='#9b59b6', width=1)), row=1, col=2)

    fig.update_layout(height=450, margin={'l': 30, 'r': 30, 't': 50, 'b': 30}, showlegend=True,
                      plot_bgcolor='#fcfcfc', paper_bgcolor='white', title_font_size=15)
    fig.update_xaxes(title_text="假阳性率 / 召回率", row=1, col=1)
    fig.update_xaxes(title_text="召回率", row=1, col=2)
    return fig


def create_user_profile_plots(df_original):
    """模块 II: 违约用户画像 (LIMIT_BAL, EDUCATION, MARRIAGE 模块 3 的关键可视化)"""
    df_plot = df_original.copy()
    df_plot['DEFAULT_STATUS'] = df_plot['DEFAULT'].map({0: '未违约', 1: '已违约'})
    df_plot['EDUCATION_LBL'] = df_plot['EDUCATION'].map({1: '研究生', 2: '大学', 3: '高中', 4: '其他'})
    df_plot['MARRIAGE_LBL'] = df_plot['MARRIAGE'].map({1: '已婚', 2: '单身', 3: '其他'})

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("信用额度 (LIMIT_BAL) 分布", "教育程度违约率", "婚姻状况违约率"))

    # 图表 1: LIMIT_BAL Box Plot
    box_fig = px.box(df_plot, x='DEFAULT_STATUS', y='LIMIT_BAL', color='DEFAULT_STATUS',
                     color_discrete_map={'未违约': '#3498db', '已违约': '#e74c3c'})
    fig.add_trace(box_fig.data[0], row=1, col=1)
    fig.add_trace(box_fig.data[1], row=1, col=1)
    fig.update_yaxes(title_text="信用额度 (NTD)", range=[0, 500000], row=1, col=1)

    # 图表 2: Education Bar Plot
    edu_rate = df_plot.groupby('EDUCATION_LBL')['DEFAULT'].mean().reset_index().sort_values(by='DEFAULT',
                                                                                            ascending=False)
    bar_edu = px.bar(edu_rate, x='EDUCATION_LBL', y='DEFAULT', color='DEFAULT', color_continuous_scale='Reds')
    fig.add_trace(bar_edu.data[0], row=1, col=2)
    fig.update_yaxes(title_text="违约率", row=1, col=2)
    fig.update_xaxes(title_text="教育程度", row=1, col=2)

    # 图表 3: Marriage Bar Plot
    marr_rate = df_plot.groupby('MARRIAGE_LBL')['DEFAULT'].mean().reset_index().sort_values(by='DEFAULT',
                                                                                            ascending=False)
    bar_marr = px.bar(marr_rate, x='MARRIAGE_LBL', y='DEFAULT', color='DEFAULT', color_continuous_scale='Reds')
    fig.add_trace(bar_marr.data[0], row=1, col=3)
    fig.update_yaxes(title_text="违约率", row=1, col=3)
    fig.update_xaxes(title_text="婚姻状况", row=1, col=3)

    fig.update_layout(height=450, showlegend=False,
                      margin={'l': 30, 'r': 30, 't': 50, 'b': 30},
                      plot_bgcolor='#fcfcfc', paper_bgcolor='white', title_font_size=15)
    return fig


def create_feature_importance_plot(importance_series):
    """模块 III: 风险解释与规则的可视化"""
    fig = px.bar(importance_series, orientation='h',
                 labels={'value': '重要性得分 (F-Score)', 'index': '特征名称'},
                 color=importance_series.index,
                 color_discrete_sequence=px.colors.qualitative.D3)
    fig.update_layout(showlegend=False, height=450,
                      yaxis={'categoryorder': 'total ascending'},
                      title='核心风险驱动因素 (XGBoost Feature Importance)',
                      margin={'l': 30, 'r': 30, 't': 50, 'b': 30},
                      plot_bgcolor='#fcfcfc', paper_bgcolor='white', title_font_size=15)
    return fig


# --- 3. Dash 应用程序布局 (集成六个模块的商业价值) ---

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server
COLOR_HEADER = '#2c3e50'
COLOR_ACCENT = '#3498db'
COLOR_ACTION = '#e74c3c'

app.layout = html.Div(style={'backgroundColor': '#f0f2f5', 'padding': '20px', 'fontFamily': 'Arial, sans-serif'},
                      children=[

                          # 顶部标题区
                          html.H1("信用卡违约风险分析仪表盘：从数据到行动",
                                  style={'textAlign': 'center', 'color': COLOR_HEADER, 'paddingBottom': '10px'}),
                          html.P(f"最佳预测模型：XGBoost | 测试集 AUC-ROC: {roc_auc:.4f}",
                                 style={'textAlign': 'center', 'color': '#34495e', 'marginBottom': '30px'}),

                          # --- 区域 I: 模型性能监控 (模块 5) ---
                          html.Div(className='row',
                                   style={'marginBottom': '30px', 'backgroundColor': 'white', 'padding': '20px',
                                          'borderRadius': '8px', 'boxShadow': '0 4px 12px 0 rgba(0,0,0,0.1)'},
                                   children=[
                                       html.H3("I. 模型性能监控 (Module 5: 评估结果)",
                                               style={'textAlign': 'center', 'color': COLOR_ACCENT}),
                                       html.P("衡量模型区分违约与非违约客户的有效性。",
                                              style={'textAlign': 'center', 'color': '#7f8c8d'}),
                                       html.Div(className='twelve columns', children=[
                                           dcc.Graph(figure=create_roc_pr_plots(y_test, xgb_probs, roc_auc))
                                       ])
                                   ]),

                          # --- 区域 II: 违约用户画像 (模块 3 & 4) ---
                          html.Div(className='row',
                                   style={'marginBottom': '30px', 'backgroundColor': 'white', 'padding': '20px',
                                          'borderRadius': '8px', 'boxShadow': '0 4px 12px 0 rgba(0,0,0,0.1)'},
                                   children=[
                                       html.H3("II. 违约用户画像与特征分布 (Module 3/4: EDA与衍生特征)",
                                               style={'textAlign': 'center', 'color': COLOR_ACCENT}),
                                       html.P("识别高风险客户的群体特征：信用额度更低、教育程度和婚姻状态也有显著差异。",
                                              style={'textAlign': 'center', 'color': '#7f8c8d'}),
                                       html.Div(className='twelve columns', children=[
                                           dcc.Graph(figure=create_user_profile_plots(df))
                                       ])
                                   ]),

                          # --- 区域 III: 模型解释与规则 (模块 6 - 放在最后) ---
                          html.Div(className='row',
                                   style={'marginBottom': '30px', 'backgroundColor': 'white', 'padding': '20px',
                                          'borderRadius': '8px', 'boxShadow': '0 4px 12px 0 rgba(0,0,0,0.1)'},
                                   children=[
                                       html.H3("III. 模型解释、行动与业务结论 (Module 6: 最终决策)",
                                               style={'textAlign': 'center', 'color': COLOR_ACTION,
                                                      'borderBottom': '2px solid ' + COLOR_ACTION,
                                                      'paddingBottom': '10px'}),

                                       html.Div(className='six columns', children=[
                                           dcc.Graph(figure=create_feature_importance_plot(top_10_xgb))
                                       ]),

                                       html.Div(className='six columns',
                                                style={'paddingTop': '20px', 'paddingLeft': '30px'}, children=[
                                               html.H3("【最终业务行动建议】", style={'color': COLOR_HEADER}),
                                               html.Ul(style={'paddingLeft': '20px', 'lineHeight': '2.0',
                                                              'listStyleType': 'none'}, children=[
                                                   html.Li([
                                                       html.Strong("1. 风险主导："),
                                                       html.Span("客户的", style={'color': '#7f8c8d'}),
                                                       html.Strong("MAX_DELAY (最大延迟月数)"),
                                                       html.Span("是预测的首要驱动因素。", style={'color': '#7f8c8d'})
                                                   ]),
                                                   html.Li([
                                                       html.Strong("2. 量化规则："),
                                                       html.Span(
                                                           f"LATE_COUNT (延迟次数) 的风险倍率达 {late_count_odds} 倍",
                                                           style={'color': '#7f8c8d'}),
                                                       html.Strong("，是制定自动化风控规则的关键依据。",
                                                                   style={'color': '#7f8c8d'})
                                                   ]),
                                                   html.Li([
                                                       html.Strong("3. 审批策略："),
                                                       html.Span("将", style={'color': '#7f8c8d'}),
                                                       html.Strong("MAX_DELAY ≥ 2"),
                                                       html.Span("和", style={'color': '#7f8c8d'}),
                                                       html.Strong("LATE_COUNT ≥ 2"),
                                                       html.Span("作为信审的硬性预警指标。", style={'color': '#7f8c8d'})
                                                   ]),
                                                   html.Li([
                                                       html.Strong("4. 贷后管理："),
                                                       html.Span("对低", style={'color': '#7f8c8d'}),
                                                       html.Strong("LIMIT_BAL"),
                                                       html.Span("且高", style={'color': '#7f8c8d'}),
                                                       html.Strong("AVG_UTILIZATION"),
                                                       html.Span("的客户群体进行重点监控和提前干预。",
                                                                 style={'color': '#7f8c8d'})
                                                   ]),
                                                   html.Li([
                                                       html.Strong("5. 模型应用："),
                                                       html.Span("使用 AUC 最高的", style={'color': '#7f8c8d'}),
                                                       html.Strong("XGBoost"),
                                                       html.Span("进行自动化风险评分和决策。",
                                                                 style={'color': '#7f8c8d'})
                                                   ]),
                                               ])
                                           ])
                                   ]),
                      ])

if __name__ == '__main__':
    # 请确保您本地运行使用 app.run() 而非 app.run_server()
    app.run(debug=True)
