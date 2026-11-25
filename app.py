"""MediChoice: interactive dashboard for the existing Variant 14 project."""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.ahp import calculate_ahp
from src.matrices import PAIRWISE_MATRICES, CRITERIA, ALTERNATIVES
from src.criteria_weights import CRITERIA_MATRIX, CRITERIA_IDS
from src.scenarios import SCENARIO_MATRICES
from src.group_decision import GROUP_NAMES, GROUP_COMPETENCE_MATRIX, GROUP_CRITERIA_MATRICES
from src.imprecise_probability import idm_bounds, contamination_bounds, robust_expected_utility

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / 'results' / 'tables'
COLORS = {'A': '#5279B7', 'B': '#148477', 'C': '#8E70B5', 'D': '#D69A40'}
COMPANIES = list(ALTERNATIVES)

st.set_page_config(page_title='MediChoice | Decision Support', page_icon='🏥', layout='wide')
st.title('MediChoice | Medical Insurance Decision Support')
st.caption('Variant 14 • AHP • Risk and uncertainty • Group decisions • Robust analysis')
st.info('Academic simulation: companies, comparison judgments, group preferences and IDM observations are hypothetical. This dashboard is not a real insurance recommendation.')

required = ['global_ranking.csv', 'criteria_weights.csv', 'priority_matrix.csv',
            'utility_matrix.csv', 'risk_uncertainty_results.csv', 'group_decision_summary.csv',
            'group_criteria_weights.csv', 'aggregation_comparison.csv', 'aggregation_winners.csv',
            'bottleneck_criteria.csv', 'decision_switching_points.csv', 'idm_robust_results.csv']
missing = [name for name in required if not (TABLES / name).is_file()]
if missing:
    st.error('Missing result files: ' + ', '.join(missing))
    st.code('python main.py\npython ranking.py\npython risk_analysis.py\npython group_analysis.py\npython aggregation_analysis.py\npython sensitivity_analysis.py', language='bash')
    st.stop()


def read(name, **kwargs):
    return pd.read_csv(TABLES / name, **kwargs)


def table(df):
    st.dataframe(df.round(6), use_container_width=True)


def winners(values):
    values = np.asarray(values, dtype=float)
    return ', '.join('Company ' + COMPANIES[i] for i in np.flatnonzero(np.isclose(values, values.max(), atol=1e-10, rtol=0)))


def bars(values, title, ylabel='Decision score'):
    df = pd.DataFrame({'Company': COMPANIES, 'Score': values})
    fig = px.bar(df, x='Company', y='Score', color='Company', color_discrete_map=COLORS,
                 text_auto='.4f', title=title)
    fig.update_layout(showlegend=False, yaxis_title=ylabel)
    st.plotly_chart(fig, use_container_width=True)


def line_chart(grid, scores, x_label, title, selected):
    df = pd.DataFrame(scores, columns=COMPANIES)
    df[x_label] = grid
    long = df.melt(id_vars=x_label, var_name='Company', value_name='Score')
    fig = px.line(long, x=x_label, y='Score', color='Company', color_discrete_map=COLORS, title=title)
    fig.add_vline(x=selected, line_dash='dash', line_color='gray')
    fig.update_xaxes(tickformat='.0%')
    st.plotly_chart(fig, use_container_width=True)


baseline = read('global_ranking.csv')
risk = read('risk_uncertainty_results.csv').set_index('Company').loc[COMPANIES]
group = read('group_decision_summary.csv').set_index('Company').loc[COMPANIES]
V = read('priority_matrix.csv', index_col=0).loc[COMPANIES, CRITERIA_IDS].to_numpy()
C = read('criteria_weights.csv').set_index('Criterion').loc[CRITERIA_IDS, 'Weight'].to_numpy()
individual_matrices = {'Baseline criteria': CRITERIA_MATRIX, **{f'Scenario: {k}': v for k, v in SCENARIO_MATRICES.items()}}
matrices = {f'{k}: {CRITERIA[k]}': (v, COMPANIES) for k, v in PAIRWISE_MATRICES.items()}
matrices.update({k: (v, CRITERIA_IDS) for k, v in individual_matrices.items()})
matrices['Group competence'] = (GROUP_COMPETENCE_MATRIX, GROUP_NAMES)
for scenario, groups in GROUP_CRITERIA_MATRICES.items():
    for name, matrix in groups.items():
        matrices[f'{scenario} / {name}'] = (matrix, CRITERIA_IDS)

page = st.sidebar.radio('Navigation', ['Overview', 'Dataset', 'AHP and matrices', 'Risk and uncertainty',
    'Group decisions', 'Linear vs Maximin', 'Sensitivity analysis', 'IDM and robustness'])
st.sidebar.caption('MediChoice • Academic decision-support simulator')

if page == 'Overview':
    cols = st.columns(4)
    cols[0].metric('Alternatives', len(COMPANIES))
    cols[1].metric('Criteria', len(CRITERIA_IDS))
    cols[2].metric('Consistent matrices', f'{sum(calculate_ahp(m)["consistent"] for m, _ in matrices.values())}/{len(matrices)}')
    cols[3].metric('Assignment probability', '2%')
    st.subheader('Baseline global ranking')
    bars(baseline.set_index('Company').loc[COMPANIES, 'Score'].to_numpy(), 'Baseline AHP')
    table(baseline)
    st.subheader('Decision summary')
    table(read('aggregation_winners.csv'))
    st.caption('Linear priorities are normalized; nonlinear Maximin scores are not probabilities.')

elif page == 'Dataset':
    st.subheader('Simulated insurance company profiles')
    path = ROOT / 'data' / 'insurance_companies.csv'
    if path.is_file():
        data = pd.read_csv(path)
        table(data)
        for col, title in [('annual_premium_rub', 'Annual premium (RUB; lower is better)'),
                           ('annual_coverage_limit_rub', 'Annual coverage limit (RUB; higher is better)')]:
            if col in data.columns and 'company_id' in data.columns:
                st.plotly_chart(px.bar(data, x='company_id', y=col, color='company_id', color_discrete_map=COLORS, title=title), use_container_width=True)
    else:
        st.warning('Dataset file is missing. Place insurance_companies.csv in the data folder.')
    table(pd.DataFrame({'Criterion': CRITERIA_IDS, 'Description': [CRITERIA[k] for k in CRITERIA_IDS],
                        'Preference': ['Minimize' if k == 'C5' else 'Maximize' for k in CRITERIA_IDS]}))
    st.caption('The fixed pairwise judgments are defined separately in Python, rather than automatically estimated from this dataset.')

elif page == 'AHP and matrices':
    name = st.selectbox('Pairwise comparison matrix', list(matrices))
    matrix, labels = matrices[name]
    result = calculate_ahp(matrix)
    table(pd.DataFrame(matrix, index=labels, columns=labels))
    cols = st.columns(4)
    cols[0].metric('Lambda max (estimate)', f'{result["lambda_max"]:.6f}')
    cols[1].metric('Consistency index', f'{result["ci"]:.6f}')
    cols[2].metric('Consistency ratio', f'{result["cr"]:.2%}')
    cols[3].metric('Validation', 'PASS' if result['consistent'] else 'FAIL')
    st.caption('Coursework threshold: CR ≤ 20%. For 2×2 reciprocal matrices, RI = 0; consistency is automatic.')
    weights = pd.DataFrame({'Item': labels, 'Weight': result['weights']})
    st.plotly_chart(px.bar(weights, x='Item', y='Weight', text_auto='.4f'), use_container_width=True)
    table(weights)
    st.subheader('Baseline criteria weights')
    table(read('criteria_weights.csv'))

elif page == 'Risk and uncertainty':
    st.subheader('Expected utility and Wald criterion')
    st.latex(r'E(A_i)=0.02U_{i,1}+0.98U_{i,2}')
    st.latex(r'W(A_i)=\min_s U_{i,s}')
    table(risk.reset_index())
    a, b = st.columns(2)
    with a:
        bars(risk['Expected Utility'], 'Expected utility')
        st.metric('Best under risk', winners(risk['Expected Utility']))
    with b:
        bars(risk['Wald Score'], 'Wald criterion')
        st.metric('Best under uncertainty', winners(risk['Wald Score']))

elif page == 'Group decisions':
    q = calculate_ahp(GROUP_COMPETENCE_MATRIX)['weights']
    st.subheader('Group competence')
    table(pd.DataFrame({'Group': GROUP_NAMES, 'Weight': q}))
    st.subheader('Criteria preferences by group')
    scenario = st.selectbox('Scenario', list(GROUP_CRITERIA_MATRICES))
    df = read('group_criteria_weights.csv')
    df = df[df['Scenario'] == scenario]
    st.plotly_chart(px.bar(df, x='Criterion', y='Weight', color='Group', barmode='group'), use_container_width=True)
    st.subheader('Combined results')
    table(group.reset_index())
    st.metric('Best group decision under risk', winners(group['Expected Utility']))

elif page == 'Linear vs Maximin':
    st.latex(r'W_L(A_i)=\sum_k V_{ik}C_k')
    st.latex(r'W_M(A_i)=\min_k(V_{ik}C_k)')
    mode = st.radio('Decision mode', ['Individual', 'Group'], horizontal=True)
    all_results = read('aggregation_comparison.csv')
    scenarios = all_results.loc[all_results['Mode'] == mode, 'Scenario'].unique().tolist()
    scenario = st.selectbox('Decision scenario', scenarios)
    df = all_results[(all_results['Mode'] == mode) & (all_results['Scenario'] == scenario)].set_index('Company').loc[COMPANIES]
    table(df.reset_index())
    a, b = st.columns(2)
    with a:
        bars(df['Linear Score'], 'Linear aggregation')
        st.metric('Linear winner', winners(df['Linear Score']))
    with b:
        bars(df['Maximin Score'], 'Nonlinear maximin')
        st.metric('Maximin winner', winners(df['Maximin Score']))
    st.caption('Separate axes are intentional: the two score scales differ. Group Maximin is computed within each group before competence-weighted aggregation.')
    bottlenecks = read('bottleneck_criteria.csv')
    if scenario in bottlenecks['Scenario'].unique():
        st.subheader('Bottleneck criteria (first minimum if tied)')
        table(bottlenecks[(bottlenecks['Mode'] == mode) & (bottlenecks['Scenario'] == scenario)])

elif page == 'Sensitivity analysis':
    st.subheader('Insured event probability')
    mode = st.radio('Decision mode', ['Individual', 'Group'], horizontal=True)
    U = (risk if mode == 'Individual' else group)[['Insured Event', 'No Insured Event']].to_numpy()
    p = st.slider('Probability of insured event (%)', 0.0, 100.0, 2.0, 0.1) / 100
    scores = U @ np.array([p, 1-p])
    st.metric('Best alternative', winners(scores))
    grid = np.linspace(0, 1, 1001)
    line_chart(grid, grid[:, None]*U[:, 0] + (1-grid[:, None])*U[:, 1], 'Probability', 'Probability sensitivity', p)
    bars(scores, 'Expected utility at selected probability')
    st.subheader('Premium weight sensitivity — individual baseline')
    i = CRITERIA_IDS.index('C5')
    w = st.slider('Premium criterion weight (%)', 0.0, 100.0, float(C[i]*100), 0.1) / 100
    mask = np.arange(len(C)) != i
    other = V[:, mask] @ (C[mask] / (1-C[i]))
    premium = V[:, i]
    scores = (1-w)*other + w*premium
    st.metric('Best alternative at selected premium weight', winners(scores))
    line_chart(grid, (1-grid[:, None])*other + grid[:, None]*premium, 'Premium weight', 'Premium sensitivity', w)
    st.caption('Other criteria weights are rescaled proportionally. Switching points below come from the existing analysis script.')
    table(read('decision_switching_points.csv'))

elif page == 'IDM and robustness':
    st.subheader('Imprecise Dirichlet Model and ε-contamination')
    st.caption('Adjust hypothetical observation counts. These are not real insurance claims.')
    mode = st.radio('Decision mode', ['Individual', 'Group'], horizontal=True)
    N = int(st.number_input('Number of observations N', min_value=1, max_value=1000000, value=100, step=1))
    n = int(st.number_input('Insured events n', min_value=0, max_value=N, value=min(2, N), step=1, key=f'events_{N}'))
    s = float(st.number_input('IDM strength s', min_value=0.1, max_value=1000.0, value=2.0, step=0.1))
    lo, hi, eps = idm_bounds(n, N, s)
    c = st.columns(3)
    c[0].metric('Lower probability', f'{lo:.2%}')
    c[1].metric('Upper probability', f'{hi:.2%}')
    c[2].metric('Epsilon', f'{eps:.2%}')
    st.latex(r'\underline{P}=\frac{n}{N+s},\quad \overline{P}=\frac{n+s}{N+s},\quad \varepsilon=\frac{s}{N+s}')
    equivalent = np.allclose([lo, hi], contamination_bounds(n/N, eps))
    st.write('IDM / contamination equivalence:', 'PASS' if equivalent else 'FAIL')
    U = (risk if mode == 'Individual' else group)[['Insured Event', 'No Insured Event']].to_numpy()
    low, high = robust_expected_utility(U[:, 0], U[:, 1], lo, hi)
    df = pd.DataFrame({'Company': COMPANIES, 'Lower Utility': low, 'Upper Utility': high})
    df['Midpoint'] = (low+high)/2
    df['Half Width'] = (high-low)/2
    fig = px.scatter(df, x='Midpoint', y='Company', error_x='Half Width', color='Company', color_discrete_map=COLORS, title='Robust expected utility intervals')
    fig.update_xaxes(title='Expected utility')
    st.plotly_chart(fig, use_container_width=True)
    table(df[['Company', 'Lower Utility', 'Upper Utility']])
    st.metric('Robust winner (maximum lower utility)', winners(low))
    grid = np.linspace(lo, hi, 101)
    line_chart(grid, grid[:, None]*U[:, 0]+(1-grid[:, None])*U[:, 1], 'Probability', 'Expected utility within IDM interval', n/N)
    st.caption('The maximum lower utility rule is distinct from the nonlinear Maximin aggregation over criteria.')
