import streamlit as st 
import numpy as np
import pandas as pd 
import plotly.express as px 
import plotly.graph_objects as go 
import joblib 
import psycopg2
from psycopg2.extensions import connection
from scipy.stats import ks_2samp
import datetime
import os

st.set_page_config(
    page_title="Customer Segmentation Online Retail", 
    layout="wide"
) 

def get_db():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    return conn 

def add_transaction(conn, cust_id, amount):
    c = conn.cursor()
    date_now = datetime.date.today()
    c.execute('INSERT INTO transaction (customer_id, created_at, amount) VALUES (%s, %s, %s)', 
              (cust_id, date_now, amount)) 
    conn.commit() 

def get_transaction_count(conn: connection):
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM transaction WHERE created_at >= TIMESTAMP '2025-12-10'")
    return c.fetchone()[0]

def get_customer_count(conn: connection):
    c = conn.cursor()
    c.execute("SELECT COUNT(DISTINCT customer_id) FROM transaction")
    return c.fetchone()[0]

def get_all_rfm(conn: connection, training: bool):
    reference_date = (datetime.date.today() + datetime.timedelta(days=1)) if not training else datetime.datetime(2025, 12, 9, 12, 50)
    return pd.read_sql(
        """
        SELECT
            customer_id,
            EXTRACT(DAY FROM %s - MAX(created_at)) AS recency,
            COUNT(*) AS frequency,
            SUM(amount) AS monetary
        FROM transaction
        WHERE created_at < %s
        GROUP BY customer_id
        """,
        conn,
        params=(reference_date,reference_date)
    ).rename(columns={
        "recency": "Recency",
        "frequency": "Frequency",
        "monetary": "Monetary"
    })

def get_customer_rfm(conn: connection, cust_id):
    c = conn.cursor()
    c.execute(
        """
        SELECT
            customer_id,
            (CURRENT_DATE - MAX(created_at)::date) AS recency_days,
            COUNT(*) AS frequency,
            SUM(amount) AS monetary
        FROM transaction
        WHERE customer_id = %s
        GROUP BY customer_id
        """,
        (cust_id,)
    )
    result = c.fetchmany(1)
    if len(result) == 0:
        return None
    return {
        "r": result[0][1],
        "f": result[0][2],
        "m": result[0][3]
    }

def get_cluster_all(scaler, model, rfm):
    rfm_scaled = scaler.transform(rfm)
    return model.predict(rfm_scaled)

def get_cluster(scaler, model, r, f, m):
    return get_cluster_all(scaler, model, [[r, f, m]])[0] 

def get_transactions(conn):
    return pd.read_sql('SELECT * FROM transaction', conn)

def show_parameter_bar(name):
    df = read_csv_cached(f"data/bar_{name}.csv")
    df["Cluster"] = df["Cluster"].map(cluster_names)

    fig = go.Figure()

    fig.add_bar(
        x=df["Cluster"],
        y=df["Min"],
        name="Min"
    )

    fig.add_bar(
        x=df["Cluster"],
        y=df["Mean"],
        name="Mean"
    )

    fig.add_bar(
        x=df["Cluster"],
        y=df["Max"],
        name="Max"
    )

    fig.update_layout(
        barmode="stack",
        title=name,
        xaxis_title="Cluster",
        yaxis_title="Value",
        legend_title="Metric",
        template="plotly_white"
    )

    st.plotly_chart(fig)

@st.cache_data
def read_csv_cached(name):
    return pd.read_csv(name)

@st.cache_data
def load():
    try:
        model = joblib.load('data/model.pkl')
        scaler = joblib.load('data/scaler.pkl')
    except:
        model = None 
        scaler = None
    return model, scaler 


cluster_names = {
    0: "Low Value",
    1: "At Risk",
    2: "New Potential",
    3: "Champions"
}

def get_recommendation(cluster):
    recommendations = {
        0: "**Low Value (Effiency):** Pelanggan ini sudah lama pergi & nilai transaksinya kecil. Jangan habiskan biaya iklan di sini. Cukup kirim email otomatis atau survei kepuasan.",
        
        1: "**At Risk (Win-Back):** Pelanggan lama yang mulai menghilang. Segera kirim kampanye 'Kami Rindu Anda' dengan diskon waktu terbatas (misal: valid 24 jam) untuk memancing transaksi.",
        
        2: "**New Potential (Onboarding):** Pelanggan baru (Recency bagus), tapi belum sering belanja. Fokus edukasi produk, tawarkan barang pelengkap (Cross-sell), dan beri insentif untuk pembelian ke-2.",
        
        3: "**Champions:** Pelanggan Sultan! Berikan layanan prioritas (VIP Access) & Reward Points. Jangan spam diskon murahan, tapi berikan apresiasi eksklusif agar tidak pindah ke kompetitor."
    }
    return recommendations.get(cluster, "Lakukan analisa lebih lanjut")

def cek_drift(data, curr_data, col):
    stat, p_value = ks_2samp(data[col], curr_data[col])
    drift = p_value < 0.05
    return drift, p_value 

def main():
    st.title("Customer Segmentation: K-Means & RFM")

    conn = get_db()
    model, scalar = load()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Dashboard", 
        "Profil Customer",
        "Simulasi Penjualan", 
        "Data Drift Detection"
    ])

    data = get_all_rfm(conn, True)
    data_new = get_all_rfm(conn, False)
    
    customer_count = get_customer_count(conn)
    transaction_len = get_transaction_count(conn)

    with tab1:
        st.info("Menampilkan data gabungan historis + transaksi baru")
        st.metric("Total Customer Base", customer_count)
        st.metric("Transaksi Baru (Setelah Training)", transaction_len) 

        data_rfm = data_new[["Recency", "Frequency", "Monetary"]].to_numpy()
        data_clusters = get_cluster_all(scalar, model, data_rfm)

        c1, c2 = st.columns(2)

        _, counts = np.unique(data_clusters, return_counts=True)

        fig = px.pie(
            names=cluster_names,
            values=counts,
            title="Jumlah Customer per Cluster"
        )
        c1.plotly_chart(fig)

        df_snake = read_csv_cached("data/snake.csv")
        df_snake["Cluster"] =  df_snake["Cluster"].map(cluster_names)
        df_snake_long = df_snake.melt(
            id_vars="Cluster",
            value_vars=["Recency", "Frequency", "Monetary"],
            var_name="Metric",
            value_name="Value"
        )

        fig = px.line(
            df_snake_long,
            x="Metric",
            y="Value",
            color="Cluster",
            markers=True,
            title="RFM Snake Plot per Cluster"
        )

        fig.update_layout(
            xaxis_title="Metric",
            yaxis_title="Value",
            legend_title="Cluster"
        )

        c2.plotly_chart(fig)

        st.subheader("Analisis RFM")

        c1, c2, c3 = st.columns(3)

        with c1:
            show_parameter_bar("Recency")
        with c2:
            show_parameter_bar("Frequency")
        with c3:
            show_parameter_bar("Monetary")
    
    with tab2:
        st.header("Cari Profil Customer")
        search_id = st.text_input("Customer ID")
        if st.button("Cari") or search_id:
            cluster = 0, 0, 0, 0
            rfm = get_customer_rfm(conn, search_id)

            if rfm is not None:
                r, f, m = rfm["r"], rfm["f"], rfm["m"]
                cluster = get_cluster(scalar, model, r, f, m)

                st.success(f"Customer ID **{search_id}**")
                st.markdown(f"### Termasuk dalam Cluster: {cluster}")

                m1, m2, m3 = st.columns(3) 

                m1.metric("Recency ", int(r))
                m2.metric("Frequency ", int(f))
                m3.metric("Monetary ", f"{m}")
                st.divider() 
                st.subheader("Rekomendasi Bisnis")
                rekomendasi = get_recommendation(int(cluster))
                st.info(rekomendasi) 
        else:
            st.warning("Tidak ditemukan") 
    with tab3:
        st.header("Simulasi Transaksi")
        c1, c2 = st.columns(2)
        with c1:
            input_cust_id = st.text_input("Input Customer ID", key="sim_id")
        with c2:
            input_amount = st.number_input("Total Belanja Barang", min_value=0, step=1) 
        
        if st.button("Simpan Transaksi & Analisa"):
            if input_cust_id and input_amount > 0:
                add_transaction(conn, input_cust_id, input_amount)
                st.success("Data berhasil disimpan")
                cust_data = data[data['Customer ID'] == input_cust_id]

                if not cust_data.empty:
                    freq_prev = cust_data.iloc[0]['Frequency']
                    mon_prev = cust_data.iloc[0]['Monetary'] 
                    recency_new = 0 
                    freq_new = freq_prev + 1 
                    mon_new = mon_prev + input_amount 
                    rfm_new = [[recency_new, freq_new, mon_new]]
                    rfm_new_scaled = scalar.transform(rfm_new)
                    new_cluster = model.predict(rfm_new_scaled)[0] 

                    st.divider()
                    colres1, colres2 = st.columns(2)
                    with colres1:
                        st.markdown("Perubahan Status")
                        st.write(f"Cluster Lama: {cust_data.iloc[0]['Cluster']}")
                        st.write(f"Cluster Baru: {new_cluster}") 

                        if cust_data.iloc[0]['Cluster'] != new_cluster:
                            st.success("Customer berpindah segmen")
                        else:
                            st.info("Segmen masih sama") 
                    with colres2:
                        st.markdown("Rekomendasi Baru")
                        st.warning(get_recommendation(new_cluster))
                else:
                    st.warning("Customer ID ini adalaha Customer Baru")
            else: 
                st.error("ID dan Total Belanja harus diisi") 
    with tab4:
        st.header("Cek Data Drift")
        
        cols = ['Recency', 'Frequency', 'Monetary']
        for col in cols:
            st.subheader(f"Pemeriksaan Variabel: {col}")
            drift_detected, p_val = cek_drift(data, data_new, col)
            c1, c2, c3 = st.columns(3)
            c1.metric("P-Value (KS-Test)", f"{p_val:.4f}")
            c2.metric("Status Drift", "Terdeteksi" if drift_detected else "Aman", delta_color="inverse" if drift_detected else "normal") 

            fig = px.histogram(data, x=col, color_discrete_sequence=['blue'], opacity=0.5, labels={'x': 'Nilai'})
            fig.add_trace(px.histogram(data_new, x=col, color_discrete_sequence=['red'], opacity=0.5)._data[0])
            fig.update_layout(barmode='overlay', title=f"Distribusi Lama (Biru) Vs Distribusi Baru (Red)")
            st.plotly_chart(fig, use_container_width=True)

            if drift_detected:
                st.error(f"Data pada kolom {col} telah berubah. Segera retraining model")
            else:
                st.success(f"Data kolom {col} masih stabil")
            st.divider()

if __name__ == "__main__":
    main()