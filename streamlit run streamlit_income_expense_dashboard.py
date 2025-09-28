import io
import calendar
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# Helper utilities
# ---------------------------

@st.cache_data
def read_csv_or_excel(uploaded_file: io.BytesIO) -> pd.DataFrame:
    """Read uploaded CSV or Excel file into a cleaned DataFrame."""
    try:
        filename = uploaded_file.name.lower()
    except Exception:
        filename = "uploaded"

    if filename.endswith(('.xls', '.xlsx')):
        # try sheet 1 or 'Sheet1' first
        try:
            df = pd.read_excel(uploaded_file, sheet_name=0)
        except Exception:
            df = pd.read_excel(uploaded_file)
    else:
        # default CSV
        df = pd.read_csv(uploaded_file)

    df = df.copy()

    # Normalize column names to simple lowercase
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c: c.lower() for c in df.columns}
    df.rename(columns=lower_map, inplace=True)

    # Common expected columns (flexibly map typical names)
    col_map = {}
    # date column
    for cand in ['date', 'transaction date', 'posted', 'time']:
        if cand in df.columns:
            col_map[cand] = 'date'
            break
    # amount column
    for cand in ['amount', 'amt', 'value', 'money']:
        if cand in df.columns:
            col_map[cand] = 'amount'
            break
    # type column (income/expense)
    for cand in ['type', 'transaction type', 'flow']:
        if cand in df.columns:
            col_map[cand] = 'type'
            break
    # category
    for cand in ['category', 'cat', 'expense category']:
        if cand in df.columns:
            col_map[cand] = 'category'
            break
    # subcategory
    for cand in ['subcategory', 'sub category', 'detail']:
        if cand in df.columns:
            col_map[cand] = 'subcategory'
            break
    # account
    for cand in ['account', 'account name', 'wallet']:
        if cand in df.columns:
            col_map[cand] = 'account'
            break

    df.rename(columns=col_map, inplace=True)

    # Parse date
    if 'date' in df.columns:
        # try multiple date formats
        df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=False)
    else:
        # if no date, create a dummy date for grouping
        df['date'] = pd.NaT

    # Clean amount: remove commas/currency if present
    if 'amount' in df.columns:
        df['amount'] = df['amount'].astype(str).str.replace(r'[\$,]', '', regex=True)
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    else:
        df['amount'] = np.nan

    # If there's no type column, infer by sign of amount
    if 'type' not in df.columns:
        df['type'] = np.where(df['amount'] >= 0, 'Income', 'Expense')
    else:
        df['type'] = df['type'].astype(str).str.title()
        # common mappings
        df['type'] = df['type'].replace({'in': 'Income', 'out': 'Expense', 'credit': 'Income', 'debit': 'Expense'})

    # Ensure category and other useful columns exist
    for c in ['category', 'subcategory', 'account', 'notes']:
        if c not in df.columns:
            df[c] = np.nan

    # Add year, month and month_name for grouping
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.strftime('%Y-%m')

    return df


@st.cache_data
def pivot_table(df: pd.DataFrame, index: list, columns: list, values: str, aggfunc: str = 'sum') -> pd.DataFrame:
    if not isinstance(index, list):
        index = [index]
    if not isinstance(columns, list):
        columns = [columns]

    table = pd.pivot_table(df, values=values, index=index if index and index != [''] else None,
                           columns=columns if columns and columns != [''] else None,
                           aggfunc=aggfunc, fill_value=0)
    # reset column names
    if isinstance(table, pd.DataFrame):
        table = table.reset_index()
    return table


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Income & Expense Dashboard", layout="wide")
st.title("Income & Expense Dashboard")
st.markdown(
    "Upload your diary of income and expense (CSV or Excel). The app will parse the data, create pivot tables and interactive charts for quick analysis.")

with st.sidebar:
    st.header("Upload / Controls")
    uploaded = st.file_uploader("Upload CSV or Excel (Sheet1)", type=['csv', 'xls', 'xlsx'])
    st.write("---")
    st.markdown("**Pivot settings**")
    default_index = st.selectbox("Default index", options=['month_name', 'category', 'account', 'type'], index=0)
    default_columns = st.selectbox("Default columns", options=['type', 'category', 'account', 'year', ''], index=0)
    agg = st.selectbox("Aggregation", options=['sum', 'mean', 'count'], index=0)
    st.write("---")
    if st.button('Use sample data'):
        uploaded = None
        st.session_state['use_sample'] = True

# sample dataset
if uploaded is None and st.session_state.get('use_sample'):
    sample = pd.DataFrame({
        'date': pd.date_range('2025-01-01', periods=120, freq='D'),
        'amount': np.random.randn(120).cumsum() + 1000,
        'category': np.random.choice(['Salary', 'Food', 'Transport', 'Rent', 'Utilities', 'Entertainment'], 120),
        'type': np.random.choice(['Income', 'Expense'], 120, p=[0.2, 0.8]),
        'account': np.random.choice(['Cash', 'Bank', 'Credit Card'], 120),
        'notes': ['sample'] * 120
    })
    df = read_csv_or_excel(io.BytesIO(sample.to_csv(index=False).encode()))
else:
    if uploaded is None:
        st.info('Upload a CSV or Excel file (sheet1). You can also click "Use sample data" in the sidebar to try the dashboard.')
        st.stop()
    df = read_csv_or_excel(uploaded)

# Main cleaning step: separate incomes & expenses
# If amounts are positive but 'type' says Expense, keep as-is; we will treat amount signless and rely on 'type'.

# Show raw preview
with st.expander("Preview uploaded data (first 100 rows)"):
    st.dataframe(df.head(100))

# Filters
col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
with col1:
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = st.date_input("Date range", value=(min_date.date() if pd.notna(min_date) else datetime.today().date(),
                                                     max_date.date() if pd.notna(max_date) else datetime.today().date()))
with col2:
    categories = ['All'] + sorted(df['category'].dropna().astype(str).unique().tolist())
    sel_cat = st.selectbox("Category", categories)
with col3:
    accounts = ['All'] + sorted(df['account'].dropna().astype(str).unique().tolist())
    sel_acc = st.selectbox("Account", accounts)
with col4:
    text_search = st.text_input("Search notes / subcategory", '')

# Apply filters
mask = pd.Series(True, index=df.index)
if date_range and len(date_range) == 2:
    start, end = date_range
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    mask &= df['date'].between(start, end)
if sel_cat != 'All':
    mask &= df['category'].astype(str) == sel_cat
if sel_acc != 'All':
    mask &= df['account'].astype(str) == sel_acc
if text_search:
    mask &= df['notes'].astype(str).str.contains(text_search, case=False, na=False) | df['subcategory'].astype(str).str.contains(text_search, case=False, na=False)

filtered = df[mask].copy()

# Ensure month_name exists
filtered['month_name'] = filtered['date'].dt.to_period('M').astype(str)

# KPIs
total_income = filtered.loc[filtered['type'] == 'Income', 'amount'].sum()
total_expense = filtered.loc[filtered['type'] == 'Expense', 'amount'].sum()
balance = total_income - total_expense

k1, k2, k3 = st.columns(3)
k1.metric("Total Income", f"{total_income:,.2f}")
k2.metric("Total Expense", f"{total_expense:,.2f}")
k3.metric("Net (Income - Expense)", f"{balance:,.2f}")

# Time series: monthly summary
monthly = filtered.groupby(['month_name', 'type'])['amount'].sum().reset_index()
monthly_pivot = monthly.pivot(index='month_name', columns='type', values='amount').fillna(0)
monthly_pivot = monthly_pivot.sort_index()

fig_ts = go.Figure()
if 'Income' in monthly_pivot.columns:
    fig_ts.add_trace(go.Bar(x=monthly_pivot.index, y=monthly_pivot['Income'], name='Income'))
if 'Expense' in monthly_pivot.columns:
    fig_ts.add_trace(go.Bar(x=monthly_pivot.index, y=monthly_pivot['Expense'], name='Expense'))
fig_ts.update_layout(barmode='group', title='Monthly Income vs Expense', xaxis_title='Month', yaxis_title='Amount')

st.plotly_chart(fig_ts, use_container_width=True)

# Category breakdown (pie/treemap)
st.subheader('Category breakdown')
cat_sum = filtered.groupby(['category', 'type'])['amount'].sum().reset_index()
cat_pivot = cat_sum.pivot(index='category', columns='type', values='amount').fillna(0)
cat_pivot['net'] = cat_pivot.get('Income', 0) - cat_pivot.get('Expense', 0)
cat_pivot = cat_pivot.reset_index().sort_values('net', ascending=False)

col_a, col_b = st.columns([1, 1])
with col_a:
    fig_cat = px.pie(cat_sum, names='category', values='amount', title='Category share (combined)')
    st.plotly_chart(fig_cat, use_container_width=True)
with col_b:
    if not cat_pivot.empty:
        st.dataframe(cat_pivot.head(50))

# Pivot table control
st.markdown('---')
st.subheader('Create a pivot table')
index_choice = st.multiselect('Index (rows)', options=['month_name', 'category', 'subcategory', 'account', 'type', 'year'], default=[default_index])
col_choice = st.multiselect('Columns', options=['type', 'category', 'account', 'year', 'month_name', ''], default=[default_columns] if default_columns else [])
val_choice = st.selectbox('Values column', options=['amount'], index=0)
agg_choice = st.selectbox('Aggregation function', options=['sum', 'mean', 'count'], index=0)

pt = pivot_table(filtered, index=index_choice, columns=[c for c in col_choice if c and c != ''], values=val_choice, aggfunc=agg_choice)

st.markdown('**Pivot result**')
st.dataframe(pt)

# Download pivot
csv = pt.to_csv(index=False).encode('utf-8')
st.download_button(label='Download pivot CSV', data=csv, file_name='pivot.csv', mime='text/csv')

# Top transactions
st.subheader('Top transactions')
num = st.slider('How many rows?', 5, 50, 10)
st.dataframe(filtered.sort_values('amount', ascending=False).head(num))

# Export filtered dataset
out_csv = filtered.to_csv(index=False).encode('utf-8')
st.download_button('Download filtered data (CSV)', data=out_csv, file_name='filtered_transactions.csv', mime='text/csv')

# Footer / tips
st.markdown('---')
st.caption('Tips: Make sure your file has a date column and an amount column. Column name matching is flexible (case insensitive).')

# End
