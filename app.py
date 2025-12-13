import streamlit as st 
import pandas as pd 
import plotly.express as px 
import plotly.graph_objects as go 
from sklearn.preprocessing import PowerTransformer
import joblib 
import sqlite3
from scipy.stats import ks_2samp
import datetime

st.set_page_config(
    page_title="Customer Segmentation Online Retail", 
    layout="wide"
) 

def init_db():
    conn = sqlite3.connect('transaction.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS new_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            transaction_date DATE,
            amount REAL
        )
    ''')
    conn.commit()
    return conn 

def add_transaction(conn, cust_id, amount):
    c = conn.cursor()
    date_now = datetime.date.today()
    c.execute('INSERT INTO new_transactions (customer_id, transaction_date, amount) VALUES (?, ?, ?)', 
              (cust_id, date_now, amount)) 
    conn.commit() 

def get_transactions(conn):
    return pd.read_sql('SELECT * FROM new_transactions', conn)

@st.cache_data
def load():
    data = pd.read_csv("clusterd.csv")
    if 'Customer ID' in data.columns:
        data['Customer ID'] = data['Customer ID'].astype(float).astype(int).astype(str)  
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
    except:
        model = None 
        scaler = None
    return data, model, scaler 


def get_recommendation(cluster):
    recommendations = {
        0: "**Lost / Low Value (Stop Budget):** Pelanggan ini sudah lama pergi & nilai transaksinya kecil. Jangan habiskan biaya iklan di sini. Cukup kirim email otomatis atau survei kepuasan.",
        
        1: "**At Risk / Hibernating (Win-Back):** Pelanggan lama yang mulai menghilang. Segera kirim kampanye 'Kami Rindu Anda' dengan diskon waktu terbatas (misal: valid 24 jam) untuk memancing transaksi.",
        
        2: "**New Potential (Onboarding):** Pelanggan baru (Recency bagus), tapi belum sering belanja. Fokus edukasi produk, tawarkan barang pelengkap (Cross-sell), dan beri insentif untuk pembelian ke-2.",
        
        3: "**Champions / VIP (Retention):** Pelanggan Sultan! Berikan layanan prioritas (VIP Access) & Reward Points. Jangan spam diskon murahan, tapi berikan apresiasi eksklusif agar tidak pindah ke kompetitor."
    }
    return recommendations.get(cluster, "Lakukan analisa lebih lanjut")

def cek_drift(data, curr_data, col):
    stat, p_value = ks_2samp(data[col], curr_data[col])
    drift = p_value < 0.05
    return drift, p_value 

def main():
    st.title("Customer Segmentation: K-Means & RFM")

    conn = init_db()
    data, model, scalar = load()

    if model is None:
        st.error("Model belum ditemukan")
        return
    
    data_new = get_transactions(conn) 
    curr_data = data.copy() 
    tab1, tab2, tab3, tab4 = st.tabs([
        "Dashboard", 
        "Profil Customer",
        "Simulasi Penjualan", 
        "Data Drift Detection"
    ])

    with tab1:
        st.info("Menampilkan data gabungan historis + transaksi baru")
        st.metric("Total Customer Base", len(data))
        st.metric("Transaksi Baru (SQLite)", len(data_new)) 
    with tab2:
        st.header("Cari Profil Customer")
        search_id = st.text_input("Customer ID")
        if st.button("Cari") or search_id:
            cust_data = curr_data[curr_data['Customer ID'] == search_id]
            if not cust_data.empty:
                cust = cust_data.iloc[0]
                cluster_id = cust['Cluster']

                st.success(f"Customer ID **{search_id}**")
                st.markdown(f"### Termasuk dalam Cluster: {cluster_id}")

                m1, m2, m3 = st.columns(3) 

                m1.metric("Recency ", int(cust['Recency']))
                m2.metric("Frequency ", int(cust['Frequency']))
                m3.metric("Monetary ", f"{cust['Monetary']:,.2f}")
                st.divider() 
                st.subheader("Rekomendasi Bisnis")
                rekomendasi = get_recommendation(int(cluster_id))
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
        
        if not data_new.empty:
            new_agg = data_new.groupby('customer_id')['amount'].sum().reset_index()
            monitoring = data.copy() 
            for index, row in new_agg.iterrows():
                i = monitoring[monitoring['Customer ID'] == row['customer_id']].index 
                if not i.empty:
                    monitoring.loc[i, 'Monetary'] += row['amount']
                    monitoring.loc[i, 'Frequency'] += 1 
                    monitoring.loc[i, 'Recency'] = 0 
        else:
            monitoring = data.copy() 
        
        cols = ['Recency', 'Frequency', 'Monetary']
        for col in cols:
            st.subheader(f"Pemeriksaan Variabel: {col}")
            drift_detected, p_val = cek_drift(data, monitoring, col)
            c1, c2, c3 = st.columns(3)
            c1.metric("P-Value (KS-Test)", f"{p_val:.4f}")
            c2.metric("Status Drift", "Terdeteksi" if drift_detected else "Aman", delta_color="inverse" if drift_detected else "normal") 

            fig = px.histogram(data, x=col, color_discrete_sequence=['blue'], opacity=0.5, labels={'x': 'Nilai'})
            fig.add_trace(px.histogram(monitoring, x=col, color_discrete_sequence=['red'], opacity=0.5)._data[0])
            fig.update_layout(barmode='overlay', title=f"Distribusi Lama (Biru) Vs Distribusi Baru (Red)")
            st.plotly_chart(fig, use_container_width=True)

            if drift_detected:
                st.error(f"Data pada kolom {col} telah berubah. Segera retraining model")
            else:
                st.success(f"Data kolom {col} masih stabil")
            st.divider()

if __name__ == "__main__":
    main()