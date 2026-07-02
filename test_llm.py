from src.llm_agent import generate_green_recommendation

siparis = {
    "Ürün": "Kraft Koli",
    "Nem Oranı": 10.5,
    "Fırın Sıcaklığı": 180,
    "Hat Hızı": 145,
    "Gramaj": 120
}

sonuc = generate_green_recommendation(
    siparis_detaylari=siparis,
    mevcut_co2_kg=285.7,
    ecoscore_harfi="C"
)

print(sonuc)