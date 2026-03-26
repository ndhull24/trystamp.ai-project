import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import warnings
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import duckdb

warnings.filterwarnings('ignore')

# --- Analytics Functions ---

def vendors_changed_banking(data):
    """Return vendors who used more than one bank account."""
    changes = data.drop_duplicates(subset=['vendor_id', 'primary_bank_account'])
    counts = changes.groupby('vendor_id')['primary_bank_account'].nunique()
    changed = counts[counts > 1].index.tolist()
    return sorted(data[data['vendor_id'].isin(changed)]['vendor'].unique().tolist())

def average_invoice(data):
    return data['invoice_amount'].mean()

def top_vendors(data, N=5):
    return data.groupby('vendor')['invoice_amount'].sum().sort_values(ascending=False).head(N)

def detect_anomalies(data):
    """Flag invoice_amount outliers using IsolationForest."""
    model = IsolationForest(contamination=0.05, random_state=42)
    amounts = data['invoice_amount'].fillna(0).values.reshape(-1, 1)
    preds = model.fit_predict(amounts)
    anomalies = data[preds == -1]
    return anomalies[['invoice_id', 'vendor_id', 'vendor', 'invoice_amount', 'date']]

def find_duplicates(data):
    """Find possible duplicate invoices (same vendor, amount, date)."""
    return data[data.duplicated(subset=['vendor_id', 'invoice_amount', 'date'], keep=False)]

def cash_flow_recommendations(data):
    recs = []
    # Early payment discounts
    discounts = data[data['discount_terms'].fillna('') != '']
    for _, row in discounts.iterrows():
        recs.append(f"Invoice {row['invoice_id']} from {row['vendor']} offers discount: {row['discount_terms']}")
    # Late payments
    if 'paid_date' in data and 'date' in data:
        days_to_pay = (data['paid_date'] - data['date']).dt.days
        late = data[days_to_pay > 30]
        for _, row in late.iterrows():
            recs.append(f"Invoice {row['invoice_id']} from {row['vendor']} was paid late ({int((row['paid_date'] - row['date']).days)} days)")
    # Top vendor report
    for vendor, spend in top_vendors(data).items():
        recs.append(f"Top vendor: {vendor} - Total paid: ${spend:,.2f}")
    return recs

def semantic_search_agent(data, question):
    # DuckDB-powered semantic search when possible, fallback to ML when necessary
    duckdb.sql("DROP VIEW IF EXISTS ap;")
    duckdb.register('ap', data)

    q = question.lower()
    if "late" in q:
        result = duckdb.sql("""
            SELECT invoice_id, vendor, invoice_amount, date, paid_date, (paid_date-date) AS days_to_pay 
            FROM ap 
            WHERE (paid_date-date) > INTERVAL 30 DAY
            ORDER BY days_to_pay DESC
            LIMIT 10
        """).to_df()
    elif "wire" in q:
        result = duckdb.sql("""
            SELECT invoice_id, vendor, invoice_amount, method
            FROM ap
            WHERE LOWER(method) = 'wire'
            LIMIT 10
        """).to_df()
    elif "discount" in q:
        result = duckdb.sql("""
            SELECT invoice_id, vendor, invoice_amount, discount_terms
            FROM ap
            WHERE discount_terms IS NOT NULL AND discount_terms <> ''
            LIMIT 10
        """).to_df()
    elif "ach" in q:
        result = duckdb.sql("""
            SELECT invoice_id, vendor, invoice_amount, method
            FROM ap
            WHERE LOWER(method) = 'ach'
            LIMIT 10
        """).to_df()
    elif "duplicate" in q or "same" in q:
        result = duckdb.sql("""
            SELECT invoice_id, vendor, invoice_amount, date
            FROM (
                SELECT *, COUNT(*) OVER(PARTITION BY vendor_id, invoice_amount, date) AS cnt
                FROM ap
            ) 
            WHERE cnt > 1
            LIMIT 10
        """).to_df()
    elif "top 5" in q and "vendor" in q:
        result = duckdb.sql("""
            SELECT vendor, SUM(invoice_amount) as total_spend
            FROM ap
            GROUP BY vendor
            ORDER BY total_spend DESC
            LIMIT 5
        """).to_df()
    else:
        # ML fallback for freeform queries
        sentences = []
        for _, row in data.iterrows():
            paid_info = row.get('paid_date', row.get('date', ''))
            paid_info = paid_info.strftime('%Y-%m-%d') if isinstance(paid_info, pd.Timestamp) else paid_info
            fact = f"Invoice {row['invoice_id']} from {row['vendor']} was paid {row.get('paid_amount', 0)} on {paid_info}."
            sentences.append(fact)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(sentences, convert_to_tensor=True)
        q_emb = model.encode([question], convert_to_tensor=True)
        hits = util.semantic_search(q_emb, embeddings, top_k=3)[0]
        return [sentences[hit['corpus_id']] for hit in hits]

    # Streamlit presentation
    out = []
    for _, row in result.iterrows():
        line = ', '.join([f"{col}: {row[col]}" for col in result.columns])
        out.append(line)
    return out

# --- Streamlit App ---

st.set_page_config(page_title="AP Finance AI Agent", layout="wide")
st.title("🧠 Accounts Payable: One-File Data Analysis")

uploaded_file = st.file_uploader("Upload your combined AP JSON data (all_data_flat.json)", type='json')
data = None

if uploaded_file:
    data = pd.read_json(uploaded_file)
    # Robust date/type conversion (for flat JSONs!)
    for col in ['date', 'paid_date', 'last_changed']:
        if col in data.columns and not np.issubdtype(data[col].dtype, np.datetime64):
            data[col] = pd.to_datetime(data[col], errors='coerce')
    st.header("👁️ Raw Data Table")
    st.dataframe(data)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏦 Vendor Banking Changes",
        "🔍 Anomalies & Duplicates",
        "💡 Cash Flow Optimization",
        "❓ Semantic Search Q&A",
        "📈 Summary & Insights"
    ])

    with tab1:
        st.subheader("Vendors Who Changed Banking Info")
        changed_names = vendors_changed_banking(data)
        st.write(changed_names)
        st.info(f"Total vendors who changed bank info: {len(changed_names)}")

    with tab2:
        st.subheader("Detected Anomalies in Invoice Amounts")
        anomalies = detect_anomalies(data)
        st.dataframe(anomalies)
        st.subheader("Detected Potential Duplicates")
        duplicates = find_duplicates(data)
        st.dataframe(duplicates)
        st.info(f"Total anomalies: {len(anomalies)} | Total duplicates: {len(duplicates)}")

    with tab3:
        st.subheader("Cash Flow Optimization Recommendations")
        for r in cash_flow_recommendations(data):
            st.write("- ", r)

    with tab4:
        st.subheader("Semantic Search")
        example_questions = [
            "Which invoices were paid late?",
            "List invoices with payment method 'Wire'",
            "Show invoices with payment discounts",
            "Which vendors changed their bank info?",
            "Who are the top 5 vendors by total payment?",
            "Which invoices have duplicates?",
            "Show anomalies in invoice amounts",
            "Which invoices were paid after 30 days?",
            "Invoices paid via ACH?",
            "List all invoices for [Vendor Name]",
        ]
        st.markdown("**Try these questions:**")
        for q in example_questions:
            if st.button(q, key=q):
                st.session_state['semantic'] = q
        question = st.text_input(
            "Ask any AP-related question (you can also click an example above):",
            key="semantic",
            value=st.session_state.get('semantic', '')
        )
        if question:
            st.write("Top relevant facts:")
            for res in semantic_search_agent(data, question):
                st.write("- ", res)

    with tab5:
        st.subheader("Quick Stats & Decision Insights")
        # Core stats
        total_spend = data['invoice_amount'].sum()
        num_invoices = len(data)
        num_vendors = data['vendor'].nunique()
        avg_invoice = average_invoice(data)
        median_invoice = data['invoice_amount'].median()
        std_invoice = data['invoice_amount'].std()
        largest_invoice = data['invoice_amount'].max()
        smallest_invoice = data['invoice_amount'].min()
        most_recent_invoice = data.iloc[data['date'].idxmax()] if 'date' in data else None

        if 'paid_date' in data and 'date' in data:
            data['days_to_pay'] = (data['paid_date'] - data['date']).dt.days
            late_payments = data[data['days_to_pay'] > 30]
            num_late = len(late_payments)
            pct_late = (num_late / num_invoices) * 100
            avg_days_to_pay = data['days_to_pay'].mean()
        else:
            num_late = pct_late = avg_days_to_pay = None

        if 'discount_terms' in data:
            discount_invoices = data[data['discount_terms'].fillna('') != '']
            num_discount = len(discount_invoices)
        else:
            num_discount = None

        if 'method' in data:
            frequent_method = data['method'].mode()[0]
        else:
            frequent_method = None

        # Display using metrics
        st.metric("Total Spend", f"${total_spend:,.2f}")
        st.metric("Number of Invoices", num_invoices)
        st.metric("Number of Vendors", num_vendors)
        st.metric("Average Invoice Amount", f"${avg_invoice:,.2f}")
        st.metric("Median Invoice Amount", f"${median_invoice:,.2f}")
        st.metric("Std. Dev. Invoice Amount", f"${std_invoice:,.2f}")
        st.metric("Largest Invoice", f"${largest_invoice:,.2f}")
        st.metric("Smallest Invoice", f"${smallest_invoice:,.2f}")
        if most_recent_invoice is not None:
            st.metric("Most Recent Invoice", f"{most_recent_invoice['invoice_id']} (${most_recent_invoice['invoice_amount']})")

        if avg_days_to_pay is not None:
            st.metric("Average Days to Pay", f"{avg_days_to_pay:.2f}")
            st.metric("Late Payments (#)", num_late)
            st.metric("% Invoices Paid Late", f"{pct_late:.2f}%")

        if num_discount is not None:
            st.metric("Invoices With Discount Terms", num_discount)

        if frequent_method:
            st.metric("Most Used Payment Method", frequent_method)

        # Top vendors by spend visualization
        st.subheader("Top 5 Vendors by Spend")
        top5 = top_vendors(data)
        st.bar_chart(top5)
        st.write(top5)

else:
    st.info("Upload your single flat AP dataset (all_data_flat.json) to get started.")

st.write("---")
st.caption("Powered by Streamlit, scikit-learn, DuckDB, and HuggingFace transformers.")
