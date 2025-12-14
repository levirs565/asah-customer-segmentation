# ASAH Customer Segmentation

Proyek ini menjembatani data mentah dan strategi bisnis lewat segmentasi pelanggan berbasis RFM dan K-Means. Tujuannya meningkatkan retensi dan profit dengan mengganti pemasaran massal ke personal. Menggunakan prinsip Pareto, fokus diarahkan secara agresif pada segmen bernilai tinggi. Solusi ini diwujudkan dalam aplikasi Streamlit terintegrasi database untuk memantau klaster, deteksi data drift, dan simulasi penjualan yang aplikatif bagi industri.

Pembuatan model ada di [Clustering.ipynb](./Clustering.ipynb)

Model terdiri dari 2 bagian yaitu scaler dan model clustering. Scaler dapat diunduh di [data/scaler.pkl](./data/scaler.pkl). Model clustering dapat diunduh di [data/model.pkl](./data/model.pkl)

## Setup Environment

Direkomendasikan membuat venv.

Install dependensi:

```sh
pip install -r requirements_all.txt
```

Unduh dataset:

```sh
chmod +x get-data.sh
./get-data.sh
```

Anda dapat mengunduh dataset secara menual di [UCI Online Retail II](https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip). Kemudian ekstreak di folder `online+retail+ii` di proyek ini.

## Menjalankan Website

Sebelum menjalankan website, buat database Postgres dan jalankan perintah SQL di [init-db.sql](./init-db.sql)

Kemudian ubah environment variabel `DATABASE_URL` ke URL ke database Postgres. Contohnya seperti ini jika menggunakan Powershell dan database lokal:

```powershell
$env:DATABASE_URL="dbname=asah user=postgres password=root"
```

Pastikan sudah menjalankan notebook. Kemudian jalankan perintah untuk mengeksport data ke Postgres (Jalnkan ini sekali saja):

```sh
python ./export_data_psql.py
```

Jalankan website:

```sh
streamlit run app.py
```
