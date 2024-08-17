import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
import plotly.express as px
import plotly.graph_objs as go
# Load data
data = pd.read_csv('dataset.csv')

# Convert 'created_at' to datetime, and drop timezone information
data['created_at'] = pd.to_datetime(data['created_at']).dt.tz_localize(None)

# Extract day names, month names, and week of the month
data['day_name'] = data['created_at'].dt.day_name()
data['month_name'] = data['created_at'].dt.strftime('%B')  # Full month name
data['month_number'] = data['created_at'].dt.month
data['year_month'] = data['created_at'].dt.to_period('M')

# Get the current month name
current_month_name = datetime.now().strftime('%B')

# Calculate the week of the month
data['week_of_month'] = data['created_at'].apply(lambda x: (x.day - 1) // 7 + 1)
data['week'] = data['month_name'] + '-Week' + data['week_of_month'].astype(str)

# Sidebar slicers for month and week
st.sidebar.header("Filter Data")

selected_month = st.sidebar.selectbox(
    "Select Month", 
    options=sorted(data['month_name'].unique(), key=lambda x: pd.to_datetime(x, format='%B')),
    index=sorted(data['month_name'].unique(), key=lambda x: pd.to_datetime(x, format='%B')).index(current_month_name)
)

filtered_data_month = data[data['month_name'] == selected_month]

selected_week = st.sidebar.multiselect(
    "Select Week(s)", 
    options=sorted(filtered_data_month['week'].unique()), 
    default=filtered_data_month['week'].unique()
)

# Filter data based on selections
filtered_data = filtered_data_month[filtered_data_month['week'].isin(selected_week)]

# Create tabs for different sections
tab1, tab2 = st.tabs(["Transaction Information", "Customer Information"])

with tab1:
    st.write("## Transaction Information")

    col1, col2 = st.columns(2)

    # Get current and previous month data
    current_month_data = data[data['month_name'] == selected_month]
    previous_month_data = data[data['month_number'] == (current_month_data['month_number'].unique()[0] - 1)]

    # Calculate metrics for current and previous months
    unique_customers_current = current_month_data['user_id'].nunique()
    unique_customers_previous = previous_month_data['user_id'].nunique() if not previous_month_data.empty else 0
    transactions_current = len(current_month_data)
    transactions_previous = len(previous_month_data) if not previous_month_data.empty else 0

    # Calculate percentage change
    def calculate_change(current, previous):
        if previous == 0:
            return None, None
        change = (current - previous) / previous * 100
        return change, "↑" if change > 0 else "↓"

    change_customers, arrow_customers = calculate_change(unique_customers_current, unique_customers_previous)
    change_transactions, arrow_transactions = calculate_change(transactions_current, transactions_previous)

    # Display scorecards with comparison
    with col1:
        st.metric(label="Total Customers", value=unique_customers_current, delta=f"{arrow_customers} {abs(change_customers):.2f}%" if change_customers is not None else "N/A",
                  delta_color="inverse" if change_customers is not None and change_customers < 0 else "normal")

    with col2:
        st.metric(label="Total Transactions", value=transactions_current, delta=f"{arrow_transactions} {abs(change_transactions):.2f}%" if change_transactions is not None else "N/A",
                  delta_color="inverse" if change_transactions is not None and change_transactions < 0 else "normal")

    # Bar Chart: Total Transactions by Month (Ordered by Date)
    transactions_by_month = data.groupby(['year_month', 'month_name']).size().reset_index(name='transaction_count').sort_values('year_month')

    # Highlight the current month with a different color
    highlight = alt.condition(
        alt.datum.month_name == selected_month,
        alt.value('orange'),  # Color for the selected month
        alt.value('steelblue')  # Color for other months
    )

    bar_chart_month = alt.Chart(transactions_by_month).mark_bar().encode(
        x=alt.X('month_name:N', sort=transactions_by_month['month_name'].tolist(), title='Month'),
        y=alt.Y('transaction_count:Q', title='Total Transactions'),
        color=highlight,  # Apply the color condition
        tooltip=['month_name', 'transaction_count']
    ).properties(
        width=600,
        height=400
    )

    bar_chart_month += bar_chart_month.mark_text(
        align='center',
        baseline='middle',
        dy=-10  # Move labels above the bars
    ).encode(
        text='transaction_count:Q'
    )

    st.write("## Total Transactions by Month")
    st.altair_chart(bar_chart_month, use_container_width=True)

    # Line Chart: Number of Transactions Over Time (by Week)
    st.write("## Number of Transactions Over Time (by Week)")
    transactions_by_week = filtered_data['week'].value_counts().sort_index().reset_index()
    transactions_by_week.columns = ['week', 'transaction_count']

    # Calculate the percentage change compared to the previous week
    transactions_by_week['change'] = transactions_by_week['transaction_count'].pct_change() * 100

    transactions_by_week['color'] = transactions_by_week['change'].apply(lambda x: 'green' if x > 0 else 'red')

    fig = go.Figure()

    # Loop through the data to create segments with different colors
    for i in range(len(transactions_by_week) - 1):
    # Determine the color based on whether the value increased or decreased
        color = 'green' if transactions_by_week['transaction_count'].iloc[i + 1] > transactions_by_week['transaction_count'].iloc[i] else 'red'

        # Add a trace for the line segment
        fig.add_trace(go.Scatter(
            x=[transactions_by_week['week'].iloc[i], transactions_by_week['week'].iloc[i + 1]],
            y=[transactions_by_week['transaction_count'].iloc[i], transactions_by_week['transaction_count'].iloc[i + 1]],
            mode='lines+markers+text',
            line=dict(color=color, width=3),
            marker=dict(color=color),
            text=[f"{transactions_by_week['transaction_count'].iloc[i + 1]}"],
            textposition="top center",
            textfont=dict(size=12),
            showlegend=False
        ))

# Add the last point as a marker only (no line segment after it), with a label
    fig.add_trace(go.Scatter(
        x=[transactions_by_week['week'].iloc[-1]],
        y=[transactions_by_week['transaction_count'].iloc[-1]],
        mode='markers+text',
        marker=dict(color='red' if transactions_by_week['change'].iloc[-1] < 0 else 'green'),
        text=[f"{transactions_by_week['transaction_count'].iloc[-1]}"],
        textposition="top center",
        textfont=dict(size=12),
        showlegend=False
    ))

    # Update the layout of the chart
    fig.update_layout(
    title='Number of Transactions Over Time (by Week)',
    xaxis_title='Week',
    yaxis_title='Total Transactions',
    showlegend=False,
    margin=dict(l=40, r=40, t=50, b=50),
)

    # Adjust the layout of the chart to make sure the labels fit well
    fig.update_xaxes(tickangle=-45, tickmode='linear')
    fig.update_yaxes(automargin=True)

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Bar Chart: Number of Transactions by Day of the Week (sorted by transaction count)
    st.write("## Number of Transactions by Day of the Week")
    transactions_by_day = filtered_data['day_name'].value_counts().reset_index()
    transactions_by_day.columns = ['day_name', 'transaction_count']

    # Sort by transaction count (descending order)
    transactions_by_day = transactions_by_day.sort_values(by='transaction_count', ascending=False)

    bar_chart_day = alt.Chart(transactions_by_day).mark_bar().encode(
        x=alt.X('day_name:N', title='Day of the Week', sort=transactions_by_day['day_name'].tolist()),
        y=alt.Y('transaction_count:Q', title='Total Transactions'),
        tooltip=['day_name', 'transaction_count']
    ).properties(
        width=300,
        height=300
    )

    bar_chart_day += bar_chart_day.mark_text(
        align='center',
        baseline='middle',
        dy=10,  # Move labels inside the bars
        color='white'  # Ensure the labels are visible inside the bars
    ).encode(
        text='transaction_count:Q'
    )

    st.altair_chart(bar_chart_day, use_container_width=True)

    # Summary of Insights
    st.write("## Summary")
    st.write(f"Total Filtered Records: {transactions_current}")
    st.write(f"Busiest Week: {transactions_by_week['week'].iloc[transactions_by_week['transaction_count'].idxmax()]} with {transactions_by_week['transaction_count'].max()} transactions")
    st.write(f"Busiest Day: {transactions_by_day['day_name'].iloc[transactions_by_day['transaction_count'].idxmax()]} with {transactions_by_day['transaction_count'].max()} transactions")

    # Safely access the total transactions for the selected month
    month_transaction_count = transactions_by_month[transactions_by_month['month_name'] == selected_month]['transaction_count']
    if not month_transaction_count.empty:
        st.write(f"Total Transactions in {selected_month}: {month_transaction_count.values[0]}")
    else:
        st.write(f"No transactions found for {selected_month}.")

with tab2:
    st.write("## Customer Information")

    # Customer Segmentation

    # Calculate the number of transactions per customer
    customer_transactions = data.groupby('user_id').size().reset_index(name='transaction_count')

    # Determine the date of the last transaction in the dataset
    last_transaction_date = data['created_at'].max()

    # Calculate the recency for each customer (days since last purchase)
    customer_last_purchase = data.groupby('user_id')['created_at'].max().reset_index()
    customer_last_purchase['recency'] = (last_transaction_date - customer_last_purchase['created_at']).dt.days

    # Merge transaction count and recency data
    customer_data = pd.merge(customer_transactions, customer_last_purchase[['user_id', 'recency']], on='user_id')

    # Define thresholds for segmentation
    new_customer_threshold = 30  # Days since last purchase
    loyal_customer_threshold = 3  # Minimum number of transactions to be considered loyal

    # Segment customers
    def segment_customer(row):
        if row['recency'] <= new_customer_threshold:
            return 'new_customer'
        elif row['transaction_count'] >= loyal_customer_threshold:
            return 'loyal'
        else:
            return 'regular_customer'

    customer_data['segment'] = customer_data.apply(segment_customer, axis=1)

    # Create the pie chart for customer segmentation
    segmentation_counts = customer_data['segment'].value_counts().reset_index()
    segmentation_counts.columns = ['segment', 'count']

    segmentation_counts['percentage'] = (segmentation_counts['count'] / segmentation_counts['count'].sum()) * 100

    # Create the pie chart for customer segmentation with percentages
    fig = px.pie(segmentation_counts, 
                 values='count', 
                 names='segment', 
                 title='Customer Segmentation', # Add a hole in the center for a donut chart
                 labels={'segment': 'Customer Segment'}
                )

    # Customize the text inside the pie chart segments
    fig.update_traces(textposition='inside', textinfo='percent+label')

    # Disable zoom, panning, and reset on the Plotly chart
    config = {
        'scrollZoom': False,
        'displayModeBar': False
    }

    # Display the pie chart in Streamlit
    st.plotly_chart(fig, use_container_width=True, config=config)
