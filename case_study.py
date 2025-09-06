# case_study_analysis.py
# amaç: Hastaların tedavi ve uygulama süreleri üzerinde veri analizi ve tahmin çalışması yapmaya çalıştım
# hazırlayan: Habibe Bayındır
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# öncelikle excel dosyasını yükleyerek başlıyoruz
df = pd.read_excel(
    r"C:\Users\Sony\OneDrive\Masaüstü\yproje\Talent_Academy_Case_DT_2025.xlsx")

# Veride ön temizlik yapmamız gerekiyor
# Tekrar eden satırları silelim
df = df.drop_duplicates()

# Eksik kategorik  değerleri bilinmiyor ile doldurdum
categorical_cols = ["Cinsiyet", "KanGrubu", "KronikHastalik", "Bolum", "Alerji", "Tanilar"]
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Bilinmiyor")
# eksik sayısal değerleri medyan ile doldurdum
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Sayısallaştırma
df["TedaviSuresi"] = df["TedaviSuresi"].astype(str).str.extract(r'(\d+)').astype(float)
df["UygulamaSuresi"] = df["UygulamaSuresi"].astype(str).str.extract(r'(\d+)').astype(float)

# Temel İstatistikler
print("=== Tedavi Süresi İstatistikleri ===")
print(df["TedaviSuresi"].describe())
print("\n=== Uygulama Süresi İstatistikleri ===")
print(df["UygulamaSuresi"].describe())

#  Hasta Bazlı Analiz
# hasta başına toplam seans
hasta_seans = df.groupby("HastaNo")["TedaviSuresi"].sum().reset_index()
hasta_seans.columns = ["HastaNo", "ToplamSeans"]
# hasta başına ortalama uygulama süresi
hasta_uygulama = df.groupby("HastaNo")["UygulamaSuresi"].mean().reset_index()
hasta_uygulama.columns = ["HastaNo", "OrtalamaUygulama"]
print("\n=== İlk 5 Hasta Seans Bilgisi ===")
print(hasta_seans.head())

# Görselleştirmeler
plt.figure(figsize=(8,5))
sns.histplot(hasta_seans["ToplamSeans"], bins=20, kde=True)
plt.title("Hastaların Toplam Seans Dağılımı")
plt.xlabel("Toplam Seans")
plt.ylabel("Hasta Sayısı")
plt.show()
plt.figure(figsize=(8,5))
sns.boxplot(x=hasta_seans["ToplamSeans"])
plt.title("Toplam Seans Boxplot")
plt.show()

top10 = hasta_seans.sort_values(by="ToplamSeans", ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(x="ToplamSeans", y="HastaNo", data=top10, palette="viridis")
plt.title("En Çok Seans Alan 10 Hasta")
plt.show()

# tedavi süresi ve uygulama süresi scatter
plt.figure(figsize=(8,5))
plt.scatter(df["TedaviSuresi"], df["UygulamaSuresi"], alpha=0.5, color="purple")
plt.title("Tedavi Süresi vs Uygulama Süresi")
plt.xlabel("Tedavi Süresi (Seans)")
plt.ylabel("Uygulama Süresi (Dakika)")
plt.show()
# Korelasyon matrisi
corr = df[['TedaviSuresi', 'UygulamaSuresi']].corr()
print("Korelasyon Matrisi:\n", corr)

# Görselleştirme (heatmap)
plt.figure(figsize=(5,4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Tedavi Süresi - Uygulama Süresi Korelasyonu")
plt.show()
# Model eğitimi, değerlendirme ve residual analizi (tam ve doğru blok) 
# Özellik setini hazırladık 
features = ['TedaviSuresi', 'Yas', 'Cinsiyet', 'Bolum', 'KronikHastalik', 'KanGrubu']
feats = [c for c in features if c in df.columns]

# One-hot encode ile X oluştur (değişken adını X_feat koyduk, X ile karışmasın)
X_feat = pd.get_dummies(df[feats], drop_first=True)
y = df['UygulamaSuresi']

#  Train/test split - burada X_train, X_test gerçekten oluşturulacak
X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.2, random_state=42)

# Bu blokta temel lineer regresyon denendi, fakat sonuçlar düşük R² verdi.
model = LinearRegression()
model.fit(X_train, y_train)
#  Tahminler (train ve test için)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

#  Performans metrikleri
print("Train R²:", r2_score(y_train, y_pred_train))
print("Test R²:", r2_score(y_test, y_pred_test))

mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
print("Test RMSE:", rmse_test)

#  Residual analizi (TEST seti üzerinden)
residuals = y_test - y_pred_test

plt.figure(figsize=(6,4))
plt.scatter(y_pred_test, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot (Test Set)")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution (Test Set)")
plt.show()
# Daha karmaşık yapıları yakalayabilmek için RandomForest modeli denendi.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Tahmin
y_pred_rf = rf_model.predict(X_test)

# Performans
print("\n=== RandomForest Modeli ===")
print("Test R²:", r2_score(y_test, y_pred_rf))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("\n=== Lineer Regresyon (Çoklu Özelliklerle) ===")
print("Train R²:", r2_score(y_train, y_pred_train))
print("Test R²:", r2_score(y_test, y_pred_test))
print("\n>>> Genel Yorum: Lineer Regresyon düşük performans verdi, fakat RandomForest modeli daha esnek olmasına rağmen yine istenilen seviyeye ulaşamadı. Bu, veri setindeki ilişkilerin karmaşık veya zayıf olabileceğini gösteriyor. Daha fazla özellik mühendisliği ile performans geliştirilebilir.")





