"""
Bashorat
                  Sog'lom  Kasal
Haqiqat Sog'lom    TN      FP      <- FP = noto'g'ri "kasal" dedi (xavotir)
        Kasal      FN      TP      <- FN = kasalni "sog'lom" dedi (XAVFLI!)

TN = True Negative (sog'lomni sog'lom dedi)
TP = True Positive (kasalni kasal dedi)  
FP = False Positive (sog'lomni kasal dedi) — TYPE I ERROR
FN = False Negative (kasalni sog'lom dedi) — TYPE II ERROR
""" 

# which metrics when we use
"""
# Spam filtri:
# - FP (oddiy email'ni spam dedi) — yomon, lekin tuzatish oson
# - FN (spamni o'tkazib yubordi) — chidash mumkin
# => Precision muhimroq (FP ni kamaytirish)

# Rak diagnostikasi:
# - FP (sog'lomni kasal dedi) — qo'shimcha test, lekin halokat emas
# - FN (kasalni o'tkazib yubordi) — odam o'lishi mumkin!
# => Recall muhimroq (FN ni kamaytirish)

# Fraud detection (firibgarlik):
# - Disbalans (99.9% normal tranzaksiya)
# => F1-score yoki Precision-Recall curve

# Umumiy maqsad, disbalans yo'q:
# => Accuracy yetadi
"""