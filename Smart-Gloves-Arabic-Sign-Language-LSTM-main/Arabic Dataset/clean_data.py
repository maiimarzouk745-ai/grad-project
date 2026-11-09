import pandas as pd

# 1. قراءة الداتا الحالية
# بديل السطر رقم 4 بالمسار الكامل
df = pd.read_csv(r"C:\Users\YourUsername\Downloads\arabic_dataset\arabic_dataset\sign_language_dataset.csv")

# 🌟 خطوة الإنقاذ: تنظيف أسماء الأعمدة من أي مسافات خفية في الـ CSV
df.columns = df.columns.str.strip()

print(f"عدد الأسطر قبل التنظيف: {len(df)}")

# 2. المستشعرات الحية اللي هنظفها (الموجودة فعلياً في ملفك)
active_sensors = [
    'flex_thumb', 'flex_middle', 'flex_ring', 'flex_pinky', 
    'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'pitch', 'roll'
]

# 3. تطبيق معادلة الـ Boxplot (IQR) لتنظيف الهزات والقيم الشاذة
for col in active_sensors:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # الإبقاء على القيم المستقرة فقط
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    else:
        print(f"⚠️ تحذير: العمود {col} مش موجود في الملف!")

print(f"عدد الأسطر بعد تنظيف البوكس بلوت: {len(df)}")

# 4. حفظ الملف النظيف مكانه
df.to_csv("sign_language_dataset.csv", index=False)
print("✅ تم تنظيف الداتا وتحديث الملف بنجاح ومنع الـ KeyError!")