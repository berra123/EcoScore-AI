"""
generate_synthetic_data.py

Dentaş Kağıt Sanayi için sentetik üretim verisi üretir.

Oluşturulan dosya:
data/raw/production_data.csv

Çalıştırma:
python data/generate_synthetic_data.py
"""

import os
import random

import numpy as np
import pandas as pd

# -------------------------------------------------------
# Ayarlar
# -------------------------------------------------------

NUM_ROWS = 5000
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -------------------------------------------------------
# Ürün bilgileri
# -------------------------------------------------------

PRODUCTS = {
    "Standart Yumurta Viyolu": {
        "gramaj": (45, 55)
    },
    "Premium Meyve Viyolu": {
        "gramaj": (60, 90)
    },
    "Oluklu Mukavva Kutu": {
        "gramaj": (150, 300)
    },
    "Endüstriyel Ambalaj Viyolu": {
        "gramaj": (80, 130)
    }
}

PRODUCT_LIST = list(PRODUCTS.keys())

# -------------------------------------------------------
# Çıktı klasörü
# -------------------------------------------------------

OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "production_data.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------
# Yardımcı Fonksiyonlar
# -------------------------------------------------------

def calculate_natural_gas(
    gramaj,
    moisture,
    oven_speed,
    temperature,
    recycled_ratio,
):
    """
    Doğalgaz tüketimi

    Artar:
    - Nem
    - Gramaj
    - Fırın sıcaklığı

    Azalır:
    - Fırın hızı
    - Geri dönüşümlü kağıt oranı
    """

    base = 900

    gas = (
        base
        * (1 + 4.2 * moisture)
        * (gramaj / 100) ** 0.75
        * (temperature / 200) ** 1.20
        * (30 / oven_speed) ** 0.90
        * (1.08 - recycled_ratio) ** 0.60
    )

    noise = np.random.normal(0, gas * 0.05)

    gas += noise

    return round(max(gas, 80), 2)


def calculate_electricity(
    order_quantity,
    oven_speed,
):
    """
    Elektrik tüketimi

    Artar:
    - Sipariş adedi
    - Fırın hızı
    """

    base = (
        120
        + (order_quantity * 0.0065)
        + (oven_speed * 12)
    )

    noise = np.random.normal(0, base * 0.04)

    electricity = base + noise

    return round(max(electricity, 50), 2)


# -------------------------------------------------------
# Veri Üretimi
# -------------------------------------------------------

rows = []

for i in range(NUM_ROWS):

    product = random.choice(PRODUCT_LIST)

    gramaj_min, gramaj_max = PRODUCTS[product]["gramaj"]

    order_quantity = random.randint(10000, 100000)

    target_weight = random.randint(
        gramaj_min,
        gramaj_max,
    )

    recycled_ratio = round(
        np.random.uniform(0.70, 1.00),
        2,
    )

    moisture = round(
        np.random.uniform(0.05, 0.15),
        3,
    )

    oven_speed = round(
        np.random.uniform(15, 45),
        2,
    )

    temperature = round(
        np.random.uniform(180, 240),
        1,
    )

    natural_gas = calculate_natural_gas(
        gramaj=target_weight,
        moisture=moisture,
        oven_speed=oven_speed,
        temperature=temperature,
        recycled_ratio=recycled_ratio,
    )

    electricity = calculate_electricity(
        order_quantity=order_quantity,
        oven_speed=oven_speed,
    )

    rows.append(
        {
            "Siparis_ID": f"S-{10001 + i}",
            "Urun_Tipi": product,
            "Siparis_Adedi": order_quantity,
            "Hedef_Gramaj_g": target_weight,
            "Geri_Donusumlu_Kagit_Orani": recycled_ratio,
            "Hammadde_Nem_Orani": moisture,
            "Firin_Hizi_m_dk": oven_speed,
            "Firin_Sicakligi_C": temperature,
            "Dogalgaz_Tuketimi_m3": natural_gas,
            "Elektrik_Tuketimi_kWh": electricity,
        }
    )

# -------------------------------------------------------
# DataFrame Oluştur
# -------------------------------------------------------

df = pd.DataFrame(rows)

# -------------------------------------------------------
# CSV Kaydet
# -------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print("=" * 60)
print("Sentetik üretim verisi başarıyla oluşturuldu.")
print(f"Toplam satır sayısı : {len(df)}")
print(f"Dosya yolu          : {OUTPUT_FILE}")
print("=" * 60)

print("\nİlk 5 kayıt:\n")
print(df.head())