import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import joblib

# ===== Load CSV =====
df = pd.read_csv('flights.csv')
df = df[['OP_CARRIER','ORIGIN','DEST','DISTANCE','ARR_DEL15']].dropna()
df['ARR_DEL15'] = df['ARR_DEL15'].astype(int)

# ===== Encode categorical features =====
le_carrier = LabelEncoder()
le_origin = LabelEncoder()
le_dest = LabelEncoder()

df['OP_CARRIER_enc'] = le_carrier.fit_transform(df['OP_CARRIER'])
df['ORIGIN_enc'] = le_origin.fit_transform(df['ORIGIN'])
df['DEST_enc'] = le_dest.fit_transform(df['DEST'])

# ===== Balance dataset =====
df_majority = df[df['ARR_DEL15']==0]
df_minority = df[df['ARR_DEL15']==1]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1, random_state=42)

X = df_balanced[['OP_CARRIER_enc','ORIGIN_enc','DEST_enc','DISTANCE']]
y = df_balanced['ARR_DEL15']

# ===== Train model =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ===== Save model & encoders =====
joblib.dump(model, "model.pkl")
joblib.dump(le_carrier, "le_carrier.pkl")
joblib.dump(le_origin, "le_origin.pkl")
joblib.dump(le_dest, "le_dest.pkl")

print("✅ Model dan encoder berhasil disimpan!")
