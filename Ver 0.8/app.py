import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Streamlit sayfasını yapılandırma
st.set_page_config(
    page_title="Borsa Tahmini Uygulaması",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS ile arayüz tasarımını iyileştirme
st.markdown("""
        <style>
        /* Sidebar arka planını değiştirmek */
        [data-testid="stSidebar"] {
            background-color: #B6CBBD !important;
            border-right: 4px solid #003311;
        }
        .stButton > button {
            background-color: #118000;
            color: white;   
            border-radius: 8px;
            padding: 10px;
            border: none;
            font-size: 16px;
            font-weight: bold;
        }
        .stButton > button:hover {
            background-color: #0b5600;
            color: white;
        }
        /* Success mesajı düzenlemesi */
        [data-testid="stNotification"] {
            background-color: #84f594 !important; /* Tahmin sonrası sidebar ile uyumlu renk */
            border-left: 5px solid #118000;
            color: #003311 !important;
            font-weight: bold;
        }
        h1 {
            color: #4CAF50;
            text-align: center;
            font-family: 'Arial', sans-serif;
        }
        .stRadio label {
            font-size: 8px;
            font-weight: bold;
            color: #524528;
        }
        .stSelectbox label, .stDateInput label {
            font-weight: bold;
            color: #524528;
        }
        /* Üst margin (margin-top) kaldırma */
        .block-container {
            padding-top: 5px !important;
            margin-top: 5px !important
        }
    </style>

""", unsafe_allow_html=True)


# Başlık
st.title("📈 Borsa Tahmini Uygulaması")

# Adım 1: Borsa Seçimi
st.sidebar.header("Adım 1: Borsa Seçimi")
borsa_secimi = st.sidebar.selectbox(
    "Hangi borsa için tahmin yapmak istiyorsunuz?",
    [
        "NASDAQ 100 (^IXIC)",
        "S&P 500 (^GSPC)",
        "FTSE 100 (^FTSE)",
        "Nikkei 225 (^N225)",
        "BIST 100 (XU100.IS)",
        "CAC 40 (^FCHI)",
        "Dow Jones Industrial Average (^DJI)",

    ]
)

# Borsa koduna göre seçim
ticker_mapping = {
    "NASDAQ 100 (^IXIC)": "^IXIC",
    "S&P 500 (^GSPC)": "^GSPC",
    "FTSE 100 (^FTSE)": "^FTSE",
    "Nikkei 225 (^N225)": "^N225",
    "BIST 100 (XU100.IS)": "XU100.IS",
    "CAC 40 (^FCHI)":"^FCHI",
    "Dow Jones Industrial Average (^DJI)":"^DJI",
}
ticker_symbol = ticker_mapping[borsa_secimi]

# Adım 2: Bugünün Tarihi Girişi
st.sidebar.header("Adım 2: Bugünün Tarihi")
tarih_girisi = st.sidebar.date_input(
    "Tahmin için bugünün tarihini giriniz:",
    value=pd.Timestamp.now().date()
)

# Kullanıcı tarihi seçtiğinde end_date değişkenini güncelle ve değişiklik kontrolü yap
if "end_date" not in st.session_state or st.session_state.end_date != tarih_girisi:
    st.session_state.end_date = tarih_girisi
    st.session_state.data_loaded = False  # Veriler yeniden yüklenecek

# Tahmin butonu
if st.sidebar.button("Tahmin Yap") or not st.session_state.data_loaded:
    try:
        # 1. Model Yükleme
        ticker = yf.Ticker(ticker_symbol)
        borsa_adi = ticker.info.get('shortName', borsa_secimi)
        st.session_state.borsa_adi = borsa_adi

        current_directory = os.getcwd()
        model_path = os.path.join(current_directory, ticker_symbol + ".h5")
        model = load_model(model_path)
        st.success("✅ Model başarıyla yüklendi!")

        # 2. Veri İndirme ve Ölçeklendirme
        data = yf.download(ticker_symbol, start="2010-01-01", end=str(st.session_state.end_date))
        st.session_state.data = data
        close_prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        # 3. Gelecekteki Tahminler
        last_60_days = scaled_data[-60:]
        temp_input = list(last_60_days.flatten())
        n_days = 5
        predictions = []

        # Gelecekteki tahmin tarihleri, tarih_girisi baz alınarak başlatılıyor
        future_dates = [tarih_girisi + timedelta(days=i) for i in range(n_days)]

        for i in range(n_days):
            X_test = np.array(temp_input[-60:]).reshape(1, 60, 1) #(örnek sayısı, zaman adımı, özellik sayısı)
            predicted_value = model.predict(X_test, verbose=0)[0, 0] #verbose=0:Sessiz mod.Hiçbir bilgi yazdırılmaz.[0, 0] predict sonucutahminedilen ilk değeri almak için kullanılır.
            predictions.append(predicted_value)
            temp_input.append(predicted_value)

        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)) #Tahminleri eski hale geri döndürür
        # Tahmin sonuçlarını kaydet
        st.session_state.data_loaded = True
        st.session_state.close_prices = close_prices
        st.session_state.future_dates = future_dates
        st.session_state.predictions = predictions
    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")

# Eğer tahmin yapıldıysa, veri türü seçimi göster
if st.session_state.data_loaded:
    # Adım 3: Veri Görünümü
    st.sidebar.header("Adım 3: Veri Görünümü")
    options = [
        "Tahmin Grafiği",
        "Son 10 Gün ve Tahmin",
        "Günlük Değişim ve Volatilite",
        "Hareketli Ortalamalar ve RSI",
        "Son 1 Ay ve Hareketli Ortalama",
        "Tahmin Tablosu",
        "Tahmin Özeti"
    ]
    st.session_state.current_view = st.sidebar.radio("Hangi tür veriyi görmek istersiniz?", options)

    # Görünümler
    if st.session_state.current_view == "Tahmin Grafiği":
        st.subheader(f"{st.session_state.borsa_adi} - Gelecekteki Borsa Tahmini (Grafik)")
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(st.session_state.data.index[-60:], st.session_state.close_prices[-60:], label="Gerçek Fiyatlar")
        ax.plot(st.session_state.future_dates, st.session_state.predictions, label="Tahmin Edilen Fiyatlar", color='orange')
        ax.set_title(f"{st.session_state.borsa_adi} - Gelecekteki Borsa Tahmini")
        ax.set_xlabel("Tarih")
        ax.set_ylabel("Fiyat")
        plt.xticks(rotation=45)
        ax.legend()     # Grafikteki çizgiler için bir açıklama (legend) ekler.
        st.pyplot(fig)  # Streamlit aracılığıyla grafiği kullanıcıya gösterir.


    elif st.session_state.current_view == "Tahmin Tablosu":
        st.subheader(f"{st.session_state.borsa_adi} - Tahmin Sonuçları (Tablo)")
        try:
            prediction_table = pd.DataFrame({
                'Tarih': [date.strftime('%Y-%m-%d') for date in st.session_state.future_dates],
                'Tahmin Edilen Fiyat': st.session_state.predictions.flatten() # Tahmin edilen fiyatları tek boyutlu bir listeye dönüştürüp tabloya ekler.
            })
            st.dataframe(prediction_table)
        except Exception as e:
            st.error(f"Tablo gösteriminde bir hata oluştu: {e}")

    elif st.session_state.current_view == "Son 10 Gün ve Tahmin":
        st.subheader(f"{st.session_state.borsa_adi} - Son 10 Gün ve Gelecek Tahmin")
        try:
            filtered_data = st.session_state.data.loc[:str(st.session_state.end_date)]
            real_last_10_days = filtered_data['Close'][-10:].values
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(filtered_data.index[-10:], real_last_10_days, label="Son 10 Gün Gerçek Fiyatları", marker='o', color='blue')
            ax.plot(st.session_state.future_dates, st.session_state.predictions.flatten(), label="Gelecek Tahmin", marker='o', color='red')
            ax.set_title(f"{st.session_state.borsa_adi} - Son 10 Gün ve Gelecek Tahmin")
            ax.legend()
            ax.grid()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Görselleştirme sırasında bir hata oluştu: {e}")

    elif st.session_state.current_view == "Günlük Değişim ve Volatilite":
        st.subheader(f"{st.session_state.borsa_adi} - Günlük Değişim ve Volatilite")
        try:
            st.session_state.data['Daily Change (%)'] = st.session_state.data['Close'].pct_change() * 100
            st.session_state.data['Volatility'] = st.session_state.data['Daily Change (%)'].rolling(window=10).std()  # 10 günlük hareketli standart sapma (volatilite) hesaplar.
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(st.session_state.data.index[-60:], st.session_state.data['Daily Change (%)'][-60:], label="Günlük Fiyat Değişimi (%)", color='purple')
            ax.plot(st.session_state.data.index[-60:], st.session_state.data['Volatility'][-60:], label="10 Günlük Volatilite", color='orange', linestyle='--')
            ax.set_title(f"{st.session_state.borsa_adi} - Günlük Fiyat Değişimi ve Volatilite")
            ax.legend()
            ax.grid()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Veri analizi sırasında bir hata oluştu: {e}")

    elif st.session_state.current_view == "Hareketli Ortalamalar ve RSI":
        st.subheader(f"{st.session_state.borsa_adi} - Hareketli Ortalamalar ve RSI")
        try:
            data = st.session_state.data
            delta = data['Close'].diff(1)  #Günlük kapanış fiyatlarındaki değişiklikleri hesaplar (bugünkü fiyat - dünkü fiyat)
            gain = delta.where(delta > 0, 0)  #Pozitif değişiklikleri alır (kazanç)
            loss = -delta.where(delta < 0, 0)  #Negatif değişiklikleri alır (kayıp), diğer değerleri 0 olarak belirler.
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss #        # RSI için gerekli olan göreli güç oranını (Relative Strength) hesaplar.
            data['RSI'] = 100 - (100 / (1 + rs)) #RSI (Relative Strength Index) değerini hesaplar ve bir sütuna kaydeder
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index[-60:], data['RSI'][-60:], label="RSI (14 Gün)", color='purple')
            ax.axhline(70, color='red', linestyle='--', label="Aşırı Alım (70)")
            ax.axhline(30, color='green', linestyle='--', label="Aşırı Satım (30)")
            ax.set_title(f"{st.session_state.borsa_adi} - RSI")
            ax.legend()
            ax.grid()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"RSI hesaplaması sırasında bir hata oluştu: {e}")

    elif st.session_state.current_view == "Son 1 Ay ve Hareketli Ortalama":
        st.subheader(f"{st.session_state.borsa_adi} - Son 1 Ay ve Hareketli Ortalama")
        try:
            data = st.session_state.data 
            data['SMA_10'] = data['Close'].rolling(window=10).mean()  # 10 günlük basit hareketli ortalamayı hesaplar ve yeni bir sütun olarak ekler.
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index[-30:], data['Close'][-30:], label="Son 1 Ay Kapanış Fiyatı", color='blue')
            ax.plot(data.index[-30:], data['SMA_10'][-30:], label="10 Günlük Hareketli Ortalama", color='orange')
            ax.plot(data.index[-30:], data['SMA_20'][-30:], label="20 Günlük Hareketli Ortalama", color='green')
            ax.plot(st.session_state.future_dates, st.session_state.predictions.flatten(), label="Gelecek Tahmin", color='red', linestyle='--')
            ax.set_title(f"{st.session_state.borsa_adi} - Son 1 Ay ve Hareketli Ortalama")
            ax.legend()
            ax.grid()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Hareketli ortalama hesaplamasında bir hata oluştu: {e}")

    elif st.session_state.current_view == "Tahmin Özeti":
        st.subheader(f"{st.session_state.borsa_adi} - Gelecek Tahminlerin İstatistiksel Özeti")
        try:
            predictions = st.session_state.predictions.flatten()
            st.write("📊 Tahmin Özeti:")
            st.metric("Ortalama Fiyat", f"{np.mean(predictions):.2f}")
            st.metric("Maksimum Fiyat", f"{np.max(predictions):.2f}")
            st.metric("Minimum Fiyat", f"{np.min(predictions):.2f}")
            st.metric("Standart Sapma", f"{np.std(predictions):.2f}")
        except Exception as e:
            st.error(f"Tahmin özetinde bir hata oluştu: {e}")